# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stage 2 Fine-tuning for Tactile-Video Generation.

Full model fine-tuning with differentiated learning rates:
    - Semantic layers (und_trans, image_embedder): 2e-6 (preserve pre-trained knowledge)
    - Fusion projection + Diffusion head: 1e-5 (primary adaptation targets)
    - LLM backbone (showo): 2e-6 (gentle fine-tuning)

Resume from Stage 1 checkpoint for continued adaptation.
"""

import os
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from einops import rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import get_text_tokenizer, get_weight_type
from torch.nn.attention.flex_attention import flex_attention

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from datasets import MixedDataLoader, TactileVisualDataset
from utils import get_config, flatten_omega_conf, AverageMeter, denorm_vid, _freeze_params, path_to_llm_name

from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    bs_tactile = config.training.batch_size_tactile
    total_batch_size_per_gpu = bs_tactile * config.dataset.accumulation
    total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
    total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    weight_type = get_weight_type(config)

    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type,
                           device=accelerator.device)
    else:
        raise NotImplementedError

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path, add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path]
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    # Load from Stage 1 checkpoint
    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path, use_safetensors=False
        ).to(accelerator.device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

    # Stage 2: Differentiated parameter groups for fine-tuning
    # Controlled via optimizer param groups, not frozen_params
    _freeze_params(model, config.model.showo.frozen_params)
    # Enable gradient checkpointing on LLM backbone to reduce activation memory
    model.showo.gradient_checkpointing_enable()

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_visual_tokens += 1
        config.dataset.preprocessing.num_tactile_tokens += 1

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # Stage 2: Differentiated learning rates for different module groups
    # Group 1: Semantic layers (image_embedder_und, und_trans, position_embedding) — low LR
    # Group 2: Fusion projection + Diffusion head — higher LR (primary adaptation)
    # Group 3: LLM backbone (showo) — low LR
    ve_params = []       # Vision Encoder: image_embedder_und, und_trans, position_embedding
    proj_params = []     # Projector + Diffusion: fusion_proj, diffusion_head, time_embed
    showo_params = []    # LLM backbone
    other_params = []    # Everything else

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(x in n for x in ['image_embedder_und', 'und_trans', 'position_embedding']):
            ve_params.append(p)
        elif any(x in n for x in ['fusion_proj', 'diffusion_head', 'time_embed', 'image_embedder_gen',
                                   'diff_proj', 'time_embed_proj']):
            proj_params.append(p)
        elif 'showo' in n:
            showo_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if ve_params:
        param_groups.append({'params': ve_params, 'lr': optimizer_config.learning_rate_ve})
    if proj_params:
        param_groups.append({'params': proj_params, 'lr': optimizer_config.learning_rate_proj})
    if showo_params:
        param_groups.append({'params': showo_params, 'lr': optimizer_config.learning_rate_showo})
    if other_params:
        param_groups.append({'params': other_params})

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            param_groups if param_groups else model.parameters(),
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    logger.info(f"Optimizer param groups: {len(param_groups)} groups")
    for i, pg in enumerate(param_groups):
        logger.info(f"  Group {i}: {len(pg['params'])} params, LR={pg.get('lr', optimizer_config.learning_rate)}")

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    def create_dataloader(dataset, batch_size, collate_fn):
        if accelerator.num_processes > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=accelerator.num_processes,
                rank=accelerator.process_index, shuffle=True, drop_last=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_fn, shuffle=shuffle,
            num_workers=dataset_config.num_workers, drop_last=True,
        )
        return dataloader

    dataset = TactileVisualDataset(
        data_root=dataset_config.tactile_data_root,
        csv_path=dataset_config.tactile_csv_path,
        text_tokenizer=text_tokenizer,
        max_seq_len=preproc_config.max_seq_length,
        image_size=preproc_config.resolution,
        latent_height=preproc_config.latent_height,
        latent_width=preproc_config.latent_width,
        num_frames=dataset_config.num_frames,
        num_visual_tokens_per_frame=preproc_config.num_visual_tokens_per_frame,
        num_tactile_tokens_per_frame=preproc_config.num_tactile_tokens_per_frame,
        cond_dropout_prob=config.training.cond_dropout_prob,
        split="train",
        showo_token_ids=showo_token_ids,
        min_res=preproc_config.min_res,
    )
    train_dataloader_tactile = create_dataloader(
        dataset, config.training.batch_size_tactile, dataset.collate_fn
    )

    num_update_steps_per_epoch = len(train_dataloader_tactile)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")

            if config.model.showo.params_not_load is not None:
                params_to_delete = []
                for k in state_dict:
                    for n in config.model.showo.params_not_load:
                        if n in k:
                            params_to_delete.append(k)
                for k in params_to_delete:
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False if config.model.showo.params_not_load is not None else True)
            del state_dict

    mixed_loader = MixedDataLoader(
        loader_list=[train_dataloader_tactile],
        samp_probs=[1.0],
        accumulation=config.dataset.accumulation,
        mode="max_size_cycle"
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps - global_step,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training (Stage 2: Full Model Fine-tuning) *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=preproc_config.num_visual_tokens,
    )

    sampler = Sampler(transport)

    @torch.no_grad()
    def prepare_latents_and_labels(
            pixel_values, data_type, shape, image_masks, modality_positions, num_frames,
    ):
        """Same as Stage 1 — see train_tactile_stage_one.py for detailed comments."""
        if config.model.vae_model.type == 'wan21':
            if len(pixel_values.shape) == 4:
                pixel_values = pixel_values.unsqueeze(2)
            image_latents = vae_model.sample(pixel_values)
            recons_images = vae_model.batch_decode(image_latents)
            if pixel_values.shape[2] == 1:
                image_latents = image_latents.squeeze(2)
                recons_images = recons_images.squeeze(2)
        else:
            raise NotImplementedError

        c, h, w = image_latents.shape[1:]
        b, n = shape

        t_list, xt_list, ut_list = [], [], []
        for i, tp in enumerate(data_type):
            t_i, x0, x1 = transport.sample(image_latents[i][None])
            t_i, xt_i, ut_i = transport.path_sampler.plan(t_i, x0, x1)
            t_list.append(t_i)
            xt_list.append(xt_i)
            ut_list.append(ut_i)

        t = torch.stack(t_list, dim=0).squeeze(-1)
        xt = torch.cat(xt_list, dim=0)
        ut = torch.cat(ut_list, dim=0)

        image_latents_orig = image_latents.reshape(b, n, c, h, w).clone()
        xt = xt.reshape(b, n, c, h, w)
        ut = ut.reshape(b, n, c, h, w)
        t = t.reshape(b, n)

        for i in range(b):
            xt[i, :num_frames] = image_latents_orig[i, :num_frames].clone()
            t[i, :num_frames] = 1.0
            vis_sid, vis_len = modality_positions[i, 0]
            image_masks[i, vis_sid: vis_sid + vis_len] = 0

        xt = rearrange(xt.reshape(b, 2, num_frames, c, h, w), 'b m t c h w -> (b m) c t h w')
        ut = rearrange(ut.reshape(b, 2, num_frames, c, h, w), 'b m t c h w -> (b m) c t h w')
        image_latents = rearrange(image_latents_orig.reshape(b, 2, num_frames, c, h, w),
                                  'b m t c h w -> (b m) c t h w')
        t = t.reshape(b, 2, num_frames).reshape(b * 2, num_frames)
        t = t.mean(dim=-1)  # (B*2, num_frames) → (B*2,), one scalar per video segment

        return image_latents, t, ut, recons_images, image_masks

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in mixed_loader:
            data_time_m.update(time.time() - end)

            text_tokens = batch['text_tokens'].to(accelerator.device)
            text_labels = batch['text_labels'].to(accelerator.device)
            pixel_values = batch['images'].to(accelerator.device).to(weight_type)

            b, m, num_frames_t = pixel_values.shape[:3]
            pixel_values = rearrange(pixel_values, 'b m t c h w -> (b m t) c h w')
            batch['data_type'] = batch['data_type'] * (m * num_frames_t)

            text_masks = batch['text_masks'].to(accelerator.device)
            image_masks = batch['image_masks'].to(accelerator.device)
            modality_positions = batch['modality_positions'].to(accelerator.device)

            image_latents, t, image_labels, recons_images, image_masks = prepare_latents_and_labels(
                pixel_values, batch['data_type'], (b, m * num_frames_t),
                image_masks, modality_positions, num_frames=num_frames_t,
            )

            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), text_tokens.size(1),
                modality_positions, accelerator.device
            ).to(weight_type)

            logits, loss_ntp, loss_flow = model(
                text_tokens=text_tokens,
                image_latents=image_latents,
                t=t.to(weight_type),
                attention_mask=block_mask,
                text_masks=text_masks,
                image_masks=image_masks,
                text_labels=text_labels,
                image_labels=image_labels,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=text_tokens.size(1),
                device=accelerator.device,
            )

            avg_loss_ntp = accelerator.gather(loss_ntp.repeat(total_batch_size_per_gpu)).mean()
            avg_loss_flow = accelerator.gather(loss_flow.repeat(total_batch_size_per_gpu)).mean()
            loss = config.training.ntp_coeff * loss_ntp + config.training.flow_coeff * loss_flow

            accelerator.backward(loss.to(weight_type) / config.training.gradient_accumulation_steps)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

            if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    lr = [group["lr"] for group in optimizer.param_groups]
                    if len(lr) >= 3:
                        logs = {
                            "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr_ve": lr[0], "lr_proj": lr[1], "lr_showo": lr[2],
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val, "batch_time": batch_time_m.val,
                        }
                        lr_str = f"LR_ve: {lr[0]:.2e} LR_proj: {lr[1]:.2e} LR_showo: {lr[2]:.2e}"
                    else:
                        logs = {
                            "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr": lr[0] if lr else 0,
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val, "batch_time": batch_time_m.val,
                        }
                        lr_str = f"LR: {lr[0]:.2e}" if lr else "LR: 0"

                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Epoch: {epoch} Step: {global_step + 1} "
                        f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                        f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} {lr_str}"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()

                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, "final")

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            for removing_checkpoint in checkpoints[0:num_to_remove]:
                shutil.rmtree(os.path.join(output_dir, removing_checkpoint))

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
