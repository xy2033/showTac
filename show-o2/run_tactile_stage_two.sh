#!/bin/bash
# =============================================================================
# Stage 2: Full Model Fine-tuning
#
# All parameters unfrozen with differentiated learning rates:
#   - Semantic layers (und_trans, image_embedder):  2e-6 (preserve features)
#   - Fusion proj + Diffusion head:                 1e-5 (primary adaptation)
#   - LLM backbone (showo):                        2e-6 (gentle fine-tuning)
#
# Resume from Stage 1 final checkpoint.
# =============================================================================

# Update this path to your Stage 1 checkpoint if needed
STAGE1_CHECKPOINT="outputs/showo2-1.5b-tactile-stage-1/checkpoint-final/unwrapped_model"

accelerate launch train_tactile_stage_two.py \
    --config configs/showo2_1.5b_tactile_stage_two.yaml \
    model.showo.pretrained_model_path="${STAGE1_CHECKPOINT}" \
    dataset.params.tactile_data_root="/media/xy/Elements/tac/tacquad_gelsight_img/data_indoor" \
    dataset.params.tactile_csv_path="/media/xy/Elements/tac/tacquad_gelsight_img/contact_indoor_list_tvl.csv" \
    dataset.params.num_frames=5 \
    training.batch_size_tactile=2 \
    training.max_train_steps=10000 \
    optimizer.params.learning_rate_proj=0.00001 \
    optimizer.params.learning_rate_ve=0.000002 \
    optimizer.params.learning_rate_showo=0.000002
