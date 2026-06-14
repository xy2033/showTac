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
Stage 1 Fine-tuning for Tactile-Video Generation.

Freezes LLM backbone, semantic layers (image_embedder_und, und_trans, position_embedding).
Only trains fusion_proj, diffusion_head, and time_embed to adapt to the tactile domain.

Key design (tactile = another visual modality):
    - Visual video (img_gelsight): clean condition, t=1.0, no flow loss → provides context
    - Tactile video (gelsight): noised target, flow matching loss computed only here
    - No new tokens or embedders needed — tactile reuses the same [BOV]...[EOV] pipeline
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
from datasets.utils import format_sequence_tactile_gen

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

    # VAE model for encoding video frames into continuous latents
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type,
                           device=accelerator.device)
    else:
        raise NotImplementedError

    # Initialize Show-o2 model
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path, add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path]
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        # 从本地 pretrained checkpoint 加载权重
        # 注意: 不用 from_pretrained() 因为它会读取保存的 config.json，
        # 其中 llm_model_path 等仍是 HF Hub 路径，集群离线环境会报错。
        # 改为: 直接构建模型(使用 CLI 覆盖后的本地路径) + 手动加载 state_dict
        showo_config = dict(config.model.showo)
        model = Showo2Qwen2_5(**showo_config).to(accelerator.device)

        # 加载预训练权重
        pretrained_path = config.model.showo.pretrained_model_path
        if os.path.isdir(pretrained_path):
            weight_file = os.path.join(pretrained_path, "pytorch_model.bin")
        else:
            weight_file = pretrained_path

        if os.path.exists(weight_file):
            accelerator.print(f"Loading pretrained weights from {weight_file}")
            state_dict = torch.load(weight_file, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            accelerator.print(
                f"Loaded pretrained weights: missing_keys={len(missing_keys)}, "
                f"unexpected_keys={len(unexpected_keys)}"
            )
            if missing_keys:
                accelerator.print(f"  first missing keys: {missing_keys[:10]}")
            if unexpected_keys:
                accelerator.print(f"  first unexpected keys: {unexpected_keys[:10]}")
            del state_dict
        else:
            accelerator.print(f"Warning: pretrained weights not found at {weight_file}, using random init")
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

    # Stage 1: Freeze LLM backbone and semantic layers
    _freeze_params(model, config.model.showo.frozen_params)

    # Enable gradient checkpointing on LLM backbone to reduce activation memory
    # Critical for seq_len~7700 with 5-frame video training
    model.showo.gradient_checkpointing_enable()

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # For time embedding: prepend time embed to vision tokens
    if config.model.showo.add_time_embeds:
        latent_visual_tokens = (
            config.dataset.preprocessing.num_visual_tokens_per_frame
            * config.dataset.params.num_frames
        )
        latent_tactile_tokens = (
            config.dataset.preprocessing.num_tactile_tokens_per_frame
            * config.dataset.params.num_frames
        )
        if config.dataset.preprocessing.num_visual_tokens == latent_visual_tokens:
            config.dataset.preprocessing.num_visual_tokens += 1
        if config.dataset.preprocessing.num_tactile_tokens == latent_tactile_tokens:
            config.dataset.preprocessing.num_tactile_tokens += 1

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    optimizer_type = config.optimizer.name

    if optimizer_type == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

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

    # Single dataset: tactile-visual pairs (no need for mixed loader)
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
        num_visual_tokens=preproc_config.num_visual_tokens,
        num_tactile_tokens=preproc_config.num_tactile_tokens,
        cond_dropout_prob=config.training.cond_dropout_prob,
        split="train",
        frame_split_mode="contact_90_10",
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

    # Use MixedDataLoader with single loader for consistency
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
    logger.info("***** Running training (Stage 1: Tactile Adaptation) *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    # Flow matching transport
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
            pixel_values: Union[torch.FloatTensor, torch.LongTensor],
            data_type,
            shape,
            image_masks,
            modality_positions,
            num_frames: int,
    ):
        """
        Prepare noised latents and velocity targets for tactile training.

        For tactile_visual_data:
            - pixel_values contains (B*2*num_frames, C, H, W) frames
              First B*num_frames = visual, last B*num_frames = tactile
            - Visual video segment: t=1.0 (clean), flow loss masked
            - Tactile video segment: noised via transport, flow loss computed

        Returns:
            image_latents: (B*2, C, T, H_vae, W_vae) — per-video latents for model
            t: (B*2,) — one timestep per video segment
            image_labels: (B*2, C, T, H_vae, W_vae) — velocity targets (ut)
            recons_images: reconstructed images from VAE
            image_masks: modified masks (visual segment zeroed out)
        """
        # VAE encode: pixel_values shape (B*2*num_frames, C, H, W)
        if config.model.vae_model.type == 'wan21':
            if len(pixel_values.shape) == 4:
                pixel_values = pixel_values.unsqueeze(2)  # add T=1 dim for VAE
            image_latents = vae_model.sample(pixel_values)  # (B*2*T, 16, 1, Hv, Wv)
            recons_images = vae_model.batch_decode(image_latents)
            if pixel_values.shape[2] == 1:
                image_latents = image_latents.squeeze(2)  # (B*2*T, 16, Hv, Wv)
                recons_images = recons_images.squeeze(2)
        else:
            raise NotImplementedError

        c, h, w = image_latents.shape[1:]  # 16, Hv, Wv
        b, n = shape  # b=batch_size, n=2*num_frames (total frames per sample)

        image_latents = rearrange(
            image_latents.reshape(b, 2, num_frames, c, h, w),
            'b m t c h w -> (b m) c t h w'
        )

        # Sample one flow timestep per video segment. The model has one time embedding
        # per segment, so frame-wise timesteps would make x_t inconsistent with t.
        t_list, xt_list, ut_list = [], [], []
        for i in range(image_latents.shape[0]):
            t_i, x0_i, x1_i = transport.sample(image_latents[i][None])
            t_i, xt_i, ut_i = transport.path_sampler.plan(t_i, x0_i, x1_i)
            t_list.append(t_i)
            xt_list.append(xt_i)
            ut_list.append(ut_i)

        t = torch.cat(t_list, dim=0)
        xt = torch.cat(xt_list, dim=0)
        ut = torch.cat(ut_list, dim=0)

        # Tactile-specific: visual video = clean condition, tactile video = noised target.
        for i in range(b):
            visual_idx = i * 2
            xt[visual_idx] = image_latents[visual_idx].clone()
            t[visual_idx] = 1.0

            vis_sid, vis_len = modality_positions[i, 0]
            image_masks[i, vis_sid: vis_sid + vis_len] = 0

        return xt, t, ut, recons_images, image_masks

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

            # pixel_values shape: (B, 2, num_frames, C, H, W)
            # Flatten to (B*2*num_frames, C, H, W) for per-frame VAE encoding
            b, m, num_frames_t = pixel_values.shape[:3]
            pixel_values = rearrange(pixel_values, 'b m t c h w -> (b m t) c h w')
            # data_type is repeated per frame
            batch['data_type'] = batch['data_type'] * (m * num_frames_t)

            text_masks = batch['text_masks'].to(accelerator.device)
            image_masks = batch['image_masks'].to(accelerator.device)
            modality_positions = batch['modality_positions'].to(accelerator.device)

            # Prepare latents and labels with tactile-specific handling
            image_latents, t, image_labels, recons_images, image_masks = prepare_latents_and_labels(
                pixel_values,
                batch['data_type'],
                (b, m * num_frames_t),
                image_masks,
                modality_positions,
                num_frames=num_frames_t,
            )

            # Create omni-attention mask
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), text_tokens.size(1),
                modality_positions, accelerator.device
            ).to(weight_type)

            # Model forward
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

            # Gather losses
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

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    lr = [group["lr"] for group in optimizer.param_groups]

                    logs = {
                        "step_loss_ntp": avg_loss_ntp.item(),
                        "step_loss_flow": avg_loss_flow.item(),
                        "lr": lr[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Epoch: {epoch} Step: {global_step + 1} "
                        f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                        f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} LR: {lr[0]:0.6f}"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                # Visualize reconstructions
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    visualize_reconstruction_tactile(
                        pixel_values, recons_images, batch['texts'],
                        num_frames_t, global_step + 1,
                    )
                    generate_model_samples = config.experiment.get("generate_model_samples", False)
                    logger.info(f"generate_model_samples={generate_model_samples}")
                    if generate_model_samples:
                        try:
                            generate_tactile_samples(
                                model=model,
                                vae_model=vae_model,
                                text_tokenizer=text_tokenizer,
                                config=config,
                                global_step=global_step + 1,
                                device=accelerator.device,
                                weight_type=weight_type,
                                sampler=sampler,
                                showo_token_ids=showo_token_ids,
                                visual_latents=image_latents,
                                target_pixel_values=pixel_values,
                                captions=batch['texts'],
                                num_frames=num_frames_t,
                            )
                        except Exception as exc:
                            logger.exception("Tactile model sample generation failed.")
                            wandb.log(
                                {"Tactile Model Generation/error": str(exc)},
                                step=global_step + 1,
                            )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, "final")

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


@torch.no_grad()
def visualize_reconstruction_tactile(
        pixel_values, recons_images, captions, num_frames, global_step
):
    """Visualize original vs reconstructed frames for both visual and tactile videos."""
    logger.info("Visualizing tactile reconstructions...")

    # pixel_values: (B*2*num_frames, C, H, W) after flattening
    # Group into visual and tactile
    total = pixel_values.shape[0]
    b = total // (2 * num_frames)

    # Reshape to (B, 2, num_frames, C, H, W)
    pixel_values = pixel_values.reshape(b, 2, num_frames, *pixel_values.shape[1:])
    recons_images = recons_images.reshape(b, 2, num_frames, *recons_images.shape[1:])

    # Visualize first sample only
    # pixel_values[0, 0]: (T, C, H, W) → unsqueeze(0) → (1, T, C, H, W) → permute → (1, C, T, H, W)
    vis_orig = denorm_vid(pixel_values[0, 0].unsqueeze(0).permute(0, 2, 1, 3, 4))
    vis_recon = denorm_vid(recons_images[0, 0].unsqueeze(0).permute(0, 2, 1, 3, 4))
    tac_orig = denorm_vid(pixel_values[0, 1].unsqueeze(0).permute(0, 2, 1, 3, 4))
    tac_recon = denorm_vid(recons_images[0, 1].unsqueeze(0).permute(0, 2, 1, 3, 4))

    wandb_images = [
        wandb.Video(vis_orig, caption=f"Visual original: {captions[0]}", fps=2, format="mp4"),
        wandb.Video(vis_recon, caption=f"Visual recon: {captions[0]}", fps=2, format="mp4"),
        wandb.Video(tac_orig, caption=f"Tactile original: {captions[0]}", fps=2, format="mp4"),
        wandb.Video(tac_recon, caption=f"Tactile recon: {captions[0]}", fps=2, format="mp4"),
    ]
    wandb.log({"Tactile Reconstructions": wandb_images}, step=global_step)


def prepare_tactile_generation_batch(
        prompts,
        text_tokenizer,
        showo_token_ids,
        config,
        device,
):
    preproc_config = config.dataset.preprocessing
    max_seq_len = preproc_config.max_seq_length
    num_visual_tokens = preproc_config.num_visual_tokens
    num_tactile_tokens = preproc_config.num_tactile_tokens
    max_text_len = max_seq_len - num_visual_tokens - num_tactile_tokens - 6
    if max_text_len <= 0:
        raise ValueError(
            f"Invalid tactile generation sequence lengths: max_seq_len={max_seq_len}, "
            f"num_visual_tokens={num_visual_tokens}, num_tactile_tokens={num_tactile_tokens}"
        )

    batch_text_tokens = []
    batch_text_tokens_null = []
    batch_modality_positions = []
    batch_modality_positions_null = []

    for prompt in prompts:
        text_ids = text_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_len,
        ).input_ids
        null_text_ids = text_tokenizer(
            "",
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_len,
        ).input_ids

        text_tokens, _, modality_positions, _, _ = format_sequence_tactile_gen(
            text_tokens=text_ids,
            bos_id=showo_token_ids["bos_id"],
            eos_id=showo_token_ids["eos_id"],
            bov_id=showo_token_ids["bov_id"],
            eov_id=showo_token_ids["eov_id"],
            pad_id=text_tokenizer.pad_token_id,
            vid_pad_id=showo_token_ids["vid_pad_id"],
            num_visual_tokens=num_visual_tokens,
            num_tactile_tokens=num_tactile_tokens,
            max_seq_len=max_seq_len,
        )
        text_tokens_null, _, modality_positions_null, _, _ = format_sequence_tactile_gen(
            text_tokens=null_text_ids,
            bos_id=showo_token_ids["bos_id"],
            eos_id=showo_token_ids["eos_id"],
            bov_id=showo_token_ids["bov_id"],
            eov_id=showo_token_ids["eov_id"],
            pad_id=text_tokenizer.pad_token_id,
            vid_pad_id=showo_token_ids["vid_pad_id"],
            num_visual_tokens=num_visual_tokens,
            num_tactile_tokens=num_tactile_tokens,
            max_seq_len=max_seq_len,
        )

        batch_text_tokens.append(text_tokens)
        batch_text_tokens_null.append(text_tokens_null)
        batch_modality_positions.append(modality_positions)
        batch_modality_positions_null.append(modality_positions_null)

    return (
        torch.stack(batch_text_tokens, dim=0).to(device),
        torch.stack(batch_text_tokens_null, dim=0).to(device),
        torch.stack(batch_modality_positions, dim=0).to(device),
        torch.stack(batch_modality_positions_null, dim=0).to(device),
    )


@torch.no_grad()
def generate_tactile_samples(
        model,
        vae_model,
        text_tokenizer,
        config,
        global_step,
        device,
        weight_type,
        sampler,
        showo_token_ids,
        visual_latents,
        target_pixel_values,
        captions,
        num_frames,
):
    """Generate one tactile clip from the current training batch and log it to wandb."""
    logger.info("Generating tactile samples with the full model...")

    was_training = model.training
    try:
        model.eval()
        model_for_sampling = model.module if hasattr(model, "module") else model

        # only_denoise_last_image in the model currently denoises the final visual region
        # across the batch, so keep generation to one sample for an unambiguous check.
        prompt = captions[0]
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
            prepare_tactile_generation_batch(
                [prompt],
                text_tokenizer,
                showo_token_ids,
                config,
                device,
            )

        visual_cond = visual_latents[0:1].to(device=device, dtype=weight_type)
        z_tactile = torch.randn_like(visual_cond)
        image_latents = torch.cat([visual_cond, z_tactile], dim=0)

        guidance_scale = config.transport.guidance_scale
        if guidance_scale > 0:
            initial_latents = torch.cat([image_latents, image_latents], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
        else:
            initial_latents = image_latents
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions

        logger.info(
            "Tactile sample shapes: "
            f"latents={tuple(initial_latents.shape)}, "
            f"text_tokens={tuple(text_tokens.shape)}, "
            f"modality_positions={modality_positions.detach().cpu().tolist()}"
        )

        block_mask = omni_attn_mask_naive(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            device,
        ).to(weight_type)

        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens.size(1),
            guidance_scale=guidance_scale,
            only_denoise_last_image=True,
        )

        sample_fn = sampler.sample_ode(
            sampling_method=config.transport.sampling_method,
            num_steps=config.transport.num_inference_steps,
            atol=config.transport.atol,
            rtol=config.transport.rtol,
            reverse=config.transport.reverse,
            time_shifting_factor=config.transport.time_shifting_factor,
        )
        samples = sample_fn(initial_latents, model_for_sampling.t2i_generate, **model_kwargs)[-1]
        if guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]
        generated_tactile_latents = samples[-1:]

        generated_frames = vae_model.batch_decode(
            rearrange(generated_tactile_latents, "b c t h w -> (b t) c h w").unsqueeze(2)
        ).squeeze(2)
        generated_video = denorm_vid(
            rearrange(generated_frames, "(b t) c h w -> b c t h w", b=1, t=num_frames)
        )

        total = target_pixel_values.shape[0]
        batch_size = total // (2 * num_frames)
        target_pixel_values = target_pixel_values.reshape(
            batch_size, 2, num_frames, *target_pixel_values.shape[1:]
        )
        visual_video = denorm_vid(target_pixel_values[0, 0].unsqueeze(0).permute(0, 2, 1, 3, 4))
        target_video = denorm_vid(target_pixel_values[0, 1].unsqueeze(0).permute(0, 2, 1, 3, 4))

        wandb.log({
            "Tactile Model Generation/visual_condition": wandb.Video(
                visual_video, caption=f"Visual condition: {prompt}", fps=2, format="mp4"
            ),
            "Tactile Model Generation/tactile_target": wandb.Video(
                target_video, caption=f"Tactile target: {prompt}", fps=2, format="mp4"
            ),
            "Tactile Model Generation/tactile_generated": wandb.Video(
                generated_video, caption=f"Tactile generated: {prompt}", fps=2, format="mp4"
            ),
        }, step=global_step)
        logger.info("Logged tactile model generation videos.")
    finally:
        model.train(was_training)


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(f"Removing {len(removing_checkpoints)} old checkpoints")
            for removing_checkpoint in removing_checkpoints:
                shutil.rmtree(os.path.join(output_dir, removing_checkpoint))

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
            max_shard_size="50GB"  # Force single-file save (1.5B model ~6GB fp32)
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
