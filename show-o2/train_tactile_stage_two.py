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

from datasets import MixedDataLoader, TactileVisualDataset, TactileQADataset
from datasets.utils import format_sequence_tactile_gen
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
    bs_qa = OmegaConf.select(config.training, 'batch_size_tactile_qa', default=0)
    use_tactile_qa = OmegaConf.select(config.training, 'use_tactile_qa', default=False)
    # In concat mode each loader contributes independently; in sample-based modes
    # accumulation controls how many batches are fetched per step.
    # Only count QA batch when QA is actually enabled (use_tactile_qa=true).
    if config.dataset.mixed_loader_mode == 'concat_max_size_cycle' and use_tactile_qa:
        total_batch_size_per_gpu = bs_tactile + bs_qa
    else:
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
        allow_missing_force_head = OmegaConf.select(
            config.model.showo, "allow_missing_tactile_force_head", default=False
        )

        def _extract_missing_keys(load_error):
            message = str(load_error)
            marker = "following keys are missing:"
            if marker not in message:
                return []
            missing_section = message.split(marker, 1)[1].split("Please make sure", 1)[0]
            return [
                key.strip().strip(".")
                for key in missing_section.replace("\n", " ").split(",")
                if key.strip()
            ]

        try:
            model = Showo2Qwen2_5.from_pretrained(
                config.model.showo.pretrained_model_path,
                use_safetensors=False,
            ).to(accelerator.device)
        except ValueError as exc:
            missing_keys = _extract_missing_keys(exc)
            only_missing_force_head = (
                len(missing_keys) > 0
                and all(key.startswith("tactile_force_head.") for key in missing_keys)
            )
            if allow_missing_force_head and only_missing_force_head:
                logger.warning(
                    "Stage 1 checkpoint is missing tactile_force_head weights. "
                    "Randomly initializing tactile_force_head because "
                    "model.showo.allow_missing_tactile_force_head=True. "
                    f"Missing keys: {missing_keys}"
                )
                model = Showo2Qwen2_5.from_pretrained(
                    config.model.showo.pretrained_model_path,
                    use_safetensors=False,
                    low_cpu_mem_usage=False,
                ).to(accelerator.device)
            else:
                raise
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

    # Stage 2: Differentiated parameter groups for fine-tuning
    # Controlled via optimizer param groups, not frozen_params
    _freeze_params(model, config.model.showo.frozen_params)
    # Enable gradient checkpointing on LLM backbone to reduce activation memory
    model.showo.gradient_checkpointing_enable()

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Wan2.1 VAE encodes videos causally in temporal chunks: the first frame is
    # represented alone, then the remaining frames are grouped by four. Keep
    # Stage 2 token layout aligned with Stage 1 clip-level VAE training.
    def resolve_wan21_latent_frames(num_input_frames: int) -> int:
        if num_input_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_input_frames}")
        return 1 + (num_input_frames - 1) // 4

    latent_num_frames = (
        resolve_wan21_latent_frames(dataset_config.num_frames)
        if config.model.vae_model.type == 'wan21'
        else dataset_config.num_frames
    )
    latent_visual_tokens = (
        config.dataset.preprocessing.num_visual_tokens_per_frame
        * latent_num_frames
    )
    latent_tactile_tokens = (
        config.dataset.preprocessing.num_tactile_tokens_per_frame
        * latent_num_frames
    )
    if config.dataset.preprocessing.num_visual_tokens != latent_visual_tokens:
        accelerator.print(
            "Adjusting visual token count for clip-level VAE encoding: "
            f"{config.dataset.preprocessing.num_visual_tokens} -> {latent_visual_tokens} "
            f"(input_frames={dataset_config.num_frames}, latent_frames={latent_num_frames})"
        )
        config.dataset.preprocessing.num_visual_tokens = latent_visual_tokens
    if config.dataset.preprocessing.num_tactile_tokens != latent_tactile_tokens:
        accelerator.print(
            "Adjusting tactile token count for clip-level VAE encoding: "
            f"{config.dataset.preprocessing.num_tactile_tokens} -> {latent_tactile_tokens} "
            f"(input_frames={dataset_config.num_frames}, latent_frames={latent_num_frames})"
        )
        config.dataset.preprocessing.num_tactile_tokens = latent_tactile_tokens

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_visual_tokens += 1
        config.dataset.preprocessing.num_tactile_tokens += 1

    if accelerator.is_main_process:
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        OmegaConf.save(config, config_path)

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
        num_visual_tokens=preproc_config.num_visual_tokens,
        num_tactile_tokens=preproc_config.num_tactile_tokens,
        cond_dropout_prob=config.training.cond_dropout_prob,
        split="train",
        showo_token_ids=showo_token_ids,
        min_res=preproc_config.min_res,
    )
    train_dataloader_tactile = create_dataloader(
        dataset, config.training.batch_size_tactile, dataset.collate_fn
    )

    # Optional QA dataset for NTP supervision (gated by use_tactile_qa, computed above)
    qa_csv_path = OmegaConf.select(config.dataset.params, 'tactile_qa_csv_path', default=None)
    bs_qa = OmegaConf.select(config.training, 'batch_size_tactile_qa', default=0)
    if use_tactile_qa and qa_csv_path and bs_qa > 0:
        logger.info(f"Creating TactileQADataset from {qa_csv_path}")
        qa_dataset = TactileQADataset(
            data_root=dataset_config.tactile_data_root,
            csv_path=dataset_config.tactile_csv_path,
            tactile_qa_csv_path=qa_csv_path,
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
            cond_dropout_prob=0.0,
            split="train",
            showo_token_ids=showo_token_ids,
            min_res=preproc_config.min_res,
        )
        train_dataloader_qa = create_dataloader(
            qa_dataset, bs_qa, qa_dataset.collate_fn
        )
    else:
        train_dataloader_qa = None

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

            unwrapped_dir = f'{path}/unwrapped_model'
            index_path = None
            for index_name in ("pytorch_model.bin.index.json", "diffusion_pytorch_model.bin.index.json"):
                candidate = f'{unwrapped_dir}/{index_name}'
                if os.path.exists(candidate):
                    index_path = candidate
                    break
            if index_path is not None:
                # Sharded checkpoint: merge every shard listed in the index.
                accelerator.print(f"Resuming from sharded checkpoint {index_path}")
                with open(index_path, "r", encoding="utf-8") as index_file:
                    weight_map = json.load(index_file)["weight_map"]
                state_dict = {}
                for shard_name in sorted(set(weight_map.values())):
                    state_dict.update(
                        torch.load(f'{unwrapped_dir}/{shard_name}', map_location="cpu")
                    )
            else:
                accelerator.print(f"Resuming from checkpoint {unwrapped_dir}/pytorch_model.bin")
                state_dict = torch.load(f'{unwrapped_dir}/pytorch_model.bin', map_location="cpu")

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

    loader_list = [train_dataloader_tactile]
    if train_dataloader_qa is not None:
        loader_list.append(train_dataloader_qa)

    # Normalize samp_probs for sample-based modes (concat mode doesn't use them).
    n_loaders = len(loader_list)
    raw_probs = [1.0] * n_loaders
    if config.dataset.mixed_loader_mode != 'concat_max_size_cycle':
        total = sum(raw_probs)
        raw_probs = [p / total for p in raw_probs]

    mixed_loader = MixedDataLoader(
        loader_list=loader_list,
        samp_probs=raw_probs,
        accumulation=config.dataset.accumulation,
        mode=config.dataset.mixed_loader_mode
    )

    remaining_train_steps = config.training.max_train_steps - global_step
    warmup_steps = config.lr_scheduler.params.warmup_steps
    if warmup_steps is None:
        warmup_ratio = float(OmegaConf.select(config.lr_scheduler.params, "warmup_ratio", default=0.0) or 0.0)
        warmup_steps = int(remaining_train_steps * warmup_ratio)
        config.lr_scheduler.params.warmup_steps = warmup_steps
        logger.info(
            f"Computed lr warmup_steps={warmup_steps} from warmup_ratio={warmup_ratio} "
            f"and remaining_train_steps={remaining_train_steps}"
        )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=remaining_train_steps,
        num_warmup_steps=warmup_steps,
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

        if image_latents.dim() == 4:
            image_latents = image_latents.unsqueeze(2)

        b, num_segments = shape
        expected_segments = b * num_segments
        if image_latents.shape[0] != expected_segments:
            raise ValueError(
                f"Unexpected VAE batch size: got {image_latents.shape[0]}, "
                f"expected {expected_segments}"
            )

        c, latent_frames, h, w = image_latents.shape[1:]
        latent_tokens_per_segment = (
            latent_frames * config.dataset.preprocessing.num_visual_tokens_per_frame
        )
        expected_segment_len = latent_tokens_per_segment + int(config.model.showo.add_time_embeds)
        actual_segment_len = int(modality_positions[0, 0, 1].item())
        if actual_segment_len != expected_segment_len:
            raise ValueError(
                "Token layout does not match clip-level VAE latent shape: "
                f"modality length={actual_segment_len}, expected={expected_segment_len}, "
                f"latent_frames={latent_frames}, h={h}, w={w}"
            )

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

    def slice_batch_by_indices(batch, indices):
        batch_size = len(batch.get('data_type', []))
        if batch_size == 0 and torch.is_tensor(batch.get('text_tokens')):
            batch_size = batch['text_tokens'].shape[0]

        sliced = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.dim() > 0 and value.shape[0] == batch_size:
                    sliced[key] = value[indices]
                else:
                    sliced[key] = value
            elif isinstance(value, list) and len(value) == batch_size:
                sliced[key] = [value[i] for i in indices]
            else:
                sliced[key] = value
        return sliced

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in mixed_loader:
            data_time_m.update(time.time() - end)

            batch_data_types = batch.get('data_type', [])
            pure_gen_idx = [i for i, dt in enumerate(batch_data_types) if dt != 'tactile_qa_data']
            qa_idx = [i for i, dt in enumerate(batch_data_types) if dt == 'tactile_qa_data']
            micro_specs = []
            if pure_gen_idx:
                micro_specs.append(('pure_gen', pure_gen_idx))
            if qa_idx:
                micro_specs.append(('qa', qa_idx))
            if not micro_specs:
                micro_specs.append(('all', list(range(batch['text_tokens'].shape[0]))))

            # Split objectives into micro-steps to lower peak memory: pure-gen trains flow, QA trains NTP.
            step_loss_ntp = torch.zeros((), device=accelerator.device)
            step_loss_flow = torch.zeros((), device=accelerator.device)
            need_validation = (
                accelerator.sync_gradients
                and accelerator.is_main_process
                and (global_step + 1) % config.experiment.generate_every == 0
            )
            validation_records = {}

            for micro_name, micro_indices in micro_specs:
                micro_batch = slice_batch_by_indices(batch, micro_indices)

                text_tokens = micro_batch['text_tokens'].to(accelerator.device)
                text_labels = micro_batch['text_labels'].to(accelerator.device)
                pixel_values = micro_batch['images'].to(accelerator.device).to(weight_type)

                b, m, num_frames_t = pixel_values.shape[:3]
                pixel_values = rearrange(pixel_values, 'b m t c h w -> (b m) c t h w')
                micro_data_type = micro_batch['data_type'] * m

                text_masks = micro_batch['text_masks'].to(accelerator.device)
                image_masks = micro_batch['image_masks'].to(accelerator.device)
                modality_positions = micro_batch['modality_positions'].to(accelerator.device)

                image_latents, t, image_labels, recons_images, image_masks = prepare_latents_and_labels(
                    pixel_values, micro_data_type, (b, m),
                    image_masks, modality_positions, num_frames=num_frames_t,
                )

                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0), text_tokens.size(1),
                    modality_positions, accelerator.device
                ).to(weight_type)

                is_qa_micro = micro_name == 'qa'
                model_text_labels = text_labels if is_qa_micro else None
                model_image_labels = None if is_qa_micro else image_labels

                model_outputs = model(
                    text_tokens=text_tokens,
                    image_latents=image_latents,
                    t=t.to(weight_type),
                    attention_mask=block_mask,
                    text_masks=text_masks,
                    image_masks=image_masks,
                    text_labels=model_text_labels,
                    image_labels=model_image_labels,
                    modality_positions=modality_positions,
                    output_hidden_states=True,
                    max_seq_len=text_tokens.size(1),
                    device=accelerator.device,
                )

                if is_qa_micro:
                    logits, loss_ntp = model_outputs
                    loss_flow = torch.zeros((), device=loss_ntp.device, dtype=loss_ntp.dtype)
                    micro_loss = config.training.ntp_coeff * loss_ntp
                else:
                    logits, loss_flow = model_outputs
                    loss_ntp = torch.zeros((), device=loss_flow.device, dtype=loss_flow.dtype)
                    micro_loss = config.training.flow_coeff * loss_flow

                accelerator.backward(micro_loss.to(weight_type) / config.training.gradient_accumulation_steps)

                step_loss_flow = step_loss_flow + loss_flow.detach()
                step_loss_ntp = step_loss_ntp + loss_ntp.detach()

                if need_validation and micro_name == 'pure_gen' and 'pure_gen' not in validation_records:
                    validation_records['pure_gen'] = {
                        'visual_latents': image_latents[0:1].detach(),
                        'target_pixel_values': pixel_values.detach(),
                        'captions': [micro_batch['texts'][0]] if micro_batch.get('texts') else ["pure gen"],
                        'num_frames': num_frames_t,
                    }
                if need_validation and micro_name == 'qa' and 'qa' not in validation_records:
                    validation_records['qa'] = {
                        'batch': {
                            'text_tokens': text_tokens[0:1].detach(),
                            'text_labels': text_labels[0:1].detach(),
                            'text_masks': text_masks[0:1].detach(),
                            'image_masks': image_masks[0:1].detach(),
                            'modality_positions': modality_positions[0:1].detach(),
                            'texts': [micro_batch['texts'][0]] if micro_batch.get('texts') else ["qa"],
                        },
                        'image_latents_qa': image_latents[0:2].detach(),
                        'num_frames': num_frames_t,
                    }

                del (
                    micro_batch, text_tokens, text_labels, pixel_values, text_masks,
                    image_masks, modality_positions, image_latents, t, image_labels,
                    recons_images, block_mask, logits, loss_ntp, loss_flow, micro_loss,
                    model_outputs, model_text_labels, model_image_labels
                )

            avg_loss_ntp = accelerator.gather(step_loss_ntp.repeat(total_batch_size_per_gpu)).mean()
            avg_loss_flow = accelerator.gather(step_loss_flow.repeat(total_batch_size_per_gpu)).mean()

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

                # Generate validation images and QA results
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    try:
                        if 'pure_gen' in validation_records:
                            pure_record = validation_records['pure_gen']
                            generate_tactile_samples(
                                model=model, vae_model=vae_model,
                                text_tokenizer=text_tokenizer, config=config,
                                global_step=global_step + 1,
                                device=accelerator.device, weight_type=weight_type,
                                sampler=sampler, showo_token_ids=showo_token_ids,
                                visual_latents=pure_record['visual_latents'],
                                target_pixel_values=pure_record['target_pixel_values'],
                                captions=pure_record['captions'],
                                num_frames=pure_record['num_frames'],
                            )

                        if 'qa' in validation_records:
                            qa_record = validation_records['qa']
                            validate_qa_answers(
                                model=model,
                                text_tokenizer=text_tokenizer,
                                global_step=global_step + 1,
                                device=accelerator.device,
                                weight_type=weight_type,
                                batch=qa_record['batch'],
                                image_latents_qa=qa_record['image_latents_qa'],
                            )
                    except Exception as exc:
                        logger.exception("Validation generation failed.")

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
            safe_serialization=False,
            max_shard_size="50GB"  # Force single-file save (1.5B model ~6GB fp32)
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def prepare_tactile_generation_batch(
        prompts, text_tokenizer, showo_token_ids, config, device,
):
    """Build token batches for tactile generation sampling (CFG-compatible)."""
    preproc_config = config.dataset.preprocessing
    num_visual_tokens = preproc_config.num_visual_tokens
    num_tactile_tokens = preproc_config.num_tactile_tokens
    max_text_len = preproc_config.max_seq_length - num_visual_tokens - num_tactile_tokens - 6
    if max_text_len <= 0:
        raise ValueError("max_seq_len too short for video tokens")

    batch_text_tokens, batch_text_tokens_null = [], []
    batch_modality_positions, batch_modality_positions_null = [], []

    for prompt in prompts:
        text_ids = text_tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_text_len).input_ids
        null_ids = text_tokenizer("", add_special_tokens=False, truncation=True, max_length=max_text_len).input_ids

        for ids, bucket_tokens, bucket_positions in [
            (text_ids, batch_text_tokens, batch_modality_positions),
            (null_ids, batch_text_tokens_null, batch_modality_positions_null),
        ]:
            tokens, _, positions, _, _ = format_sequence_tactile_gen(
                text_tokens=ids,
                bos_id=showo_token_ids["bos_id"],
                eos_id=showo_token_ids["eos_id"],
                bov_id=showo_token_ids["bov_id"],
                eov_id=showo_token_ids["eov_id"],
                pad_id=text_tokenizer.pad_token_id,
                vid_pad_id=showo_token_ids["vid_pad_id"],
                num_visual_tokens=num_visual_tokens,
                num_tactile_tokens=num_tactile_tokens,
                max_seq_len=preproc_config.max_seq_length,
            )
            bucket_tokens.append(tokens)
            bucket_positions.append(positions)

    return (
        torch.stack(batch_text_tokens, dim=0).to(device),
        torch.stack(batch_text_tokens_null, dim=0).to(device),
        torch.stack(batch_modality_positions, dim=0).to(device),
        torch.stack(batch_modality_positions_null, dim=0).to(device),
    )


@torch.no_grad()
def generate_tactile_samples(
        model, vae_model, text_tokenizer, config, global_step,
        device, weight_type, sampler, showo_token_ids,
        visual_latents, target_pixel_values, captions, num_frames,
):
    """Generate tactile video from the current pure-gen batch and log to wandb."""
    logger.info("Generating tactile validation samples...")
    was_training = model.training
    try:
        model.eval()
        model_for_sampling = model.module if hasattr(model, "module") else model

        prompt = captions[0]
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
            prepare_tactile_generation_batch([prompt], text_tokenizer, showo_token_ids, config, device)

        visual_cond = visual_latents[0:1].to(device=device, dtype=weight_type)
        z_tactile = torch.randn_like(visual_cond)
        image_latents = torch.cat([visual_cond, z_tactile], dim=0)

        guidance_scale = config.transport.guidance_scale
        if guidance_scale > 0:
            initial_latents = torch.cat([image_latents, image_latents], dim=0)
            text_tokens_cfg = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions_cfg = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
        else:
            initial_latents = image_latents
            text_tokens_cfg = batch_text_tokens
            modality_positions_cfg = batch_modality_positions

        block_mask = omni_attn_mask_naive(
            text_tokens_cfg.size(0), text_tokens_cfg.size(1),
            modality_positions_cfg, device,
        ).to(weight_type)

        model_kwargs = dict(
            text_tokens=text_tokens_cfg,
            attention_mask=block_mask,
            modality_positions=modality_positions_cfg,
            output_hidden_states=True,
            max_seq_len=text_tokens_cfg.size(1),
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

        generated_frames = vae_model.batch_decode(generated_tactile_latents)
        generated_video = denorm_vid(generated_frames)

        if target_pixel_values.dim() == 5:
            batch_size = target_pixel_values.shape[0] // 2
            target_pixel_values = target_pixel_values.reshape(batch_size, 2, *target_pixel_values.shape[1:])
            visual_video = denorm_vid(target_pixel_values[0, 0].unsqueeze(0))
            target_video = denorm_vid(target_pixel_values[0, 1].unsqueeze(0))
        else:
            total = target_pixel_values.shape[0]
            batch_size = total // (2 * num_frames)
            target_pixel_values = target_pixel_values.reshape(batch_size, 2, num_frames, *target_pixel_values.shape[1:])
            visual_video = denorm_vid(target_pixel_values[0, 0].unsqueeze(0).permute(0, 2, 1, 3, 4))
            target_video = denorm_vid(target_pixel_values[0, 1].unsqueeze(0).permute(0, 2, 1, 3, 4))

        wandb.log({
            "Tactile Generation/visual_condition": wandb.Video(visual_video, caption=f"Visual: {prompt[:120]}", fps=2, format="mp4"),
            "Tactile Generation/tactile_target": wandb.Video(target_video, caption=f"Target: {prompt[:120]}", fps=2, format="mp4"),
            "Tactile Generation/tactile_generated": wandb.Video(generated_video, caption=f"Generated: {prompt[:120]}", fps=2, format="mp4"),
        }, step=global_step)
        logger.info("Logged tactile generation validation videos.")
    finally:
        model.train(was_training)


@torch.no_grad()
def validate_qa_answers(
        model, text_tokenizer, global_step,
        device, weight_type, batch, image_latents_qa,
):
    """Validate QA answer-token accuracy without logging QA-conditioned videos."""
    if batch is None:
        return
    was_training = model.training
    try:
        model.eval()

        # 1. NTP accuracy on answer tokens
        text_labels = batch['text_labels'].to(device)
        text_tokens_qa = batch['text_tokens'].to(device)
        modality_positions = batch['modality_positions'].to(device)
        image_masks = batch['image_masks'].to(device)
        image_latents = image_latents_qa.to(device=device, dtype=weight_type)

        text_masks = batch['text_masks'].to(device)
        block_mask = omni_attn_mask_naive(
            text_tokens_qa.size(0), text_tokens_qa.size(1),
            modality_positions, device,
        ).to(weight_type)

        logits, _ = model(
            text_tokens=text_tokens_qa,
            image_latents=image_latents,
            t=torch.ones(image_latents.shape[0], device=device).to(weight_type),
            attention_mask=block_mask,
            text_masks=text_masks,
            image_masks=image_masks,
            text_labels=text_labels,
            image_labels=None,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens_qa.size(1),
            device=device,
        )

        # Compute per-token accuracy on answer positions (labels != -100)
        valid_mask = text_labels[:, 1:] != -100
        if valid_mask.any():
            valid_labels = text_labels[:, 1:][valid_mask]
            if logits is not None and logits.dim() == 2:
                pred_ids_all = logits.argmax(dim=-1)
                correct = pred_ids_all == valid_labels
                qa_accuracy = correct.sum().float() / valid_labels.numel()
                pred_ids = pred_ids_all.tolist()
                gt_ids = valid_labels.tolist()
            else:
                preds = logits[:, :-1].argmax(dim=-1)
                correct = (preds == text_labels[:, 1:]) & valid_mask
                qa_accuracy = correct.sum().float() / valid_mask.sum().float()
                first_valid = valid_mask[0].nonzero(as_tuple=True)[0]
                pred_ids = preds[0][first_valid].tolist()
                gt_ids = text_labels[0, 1:][first_valid].tolist()

            # Decode a sample answer
            if len(gt_ids) > 0:
                pred_text = text_tokenizer.decode([t for t in pred_ids if t >= 0], skip_special_tokens=True)
                gt_text = text_tokenizer.decode([t for t in gt_ids if t >= 0], skip_special_tokens=True)

                wandb.log({
                    "QA/accuracy": qa_accuracy.item(),
                    "QA/predicted_answer": wandb.Html(f"<pre>Pred: {pred_text[:500]}\nGT:   {gt_text[:500]}</pre>"),
                }, step=global_step)
                logger.info(f"QA accuracy: {qa_accuracy.item():.4f} | Pred: {pred_text[:100]}...")
    except Exception as exc:
        logger.exception("QA validation failed.")
    finally:
        model.train(was_training)


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
