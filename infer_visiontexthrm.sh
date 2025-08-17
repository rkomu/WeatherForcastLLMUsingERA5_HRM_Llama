#!/bin/bash

python -m vision_text.infer_hrm_vision2text \
  --mae_ckpt checkpoints/satswinmae_epoch10.pt \
  --adapter_ckpt checkpoints_v2t/adapter_epoch54.pt \
  --files ./dataset/raw_data/**/*.nc \
  --variables u10 v10 r sp ssrd t cp \
  --date 2024-08-02 \
  --time_start "2024-08-02" \
  --time_end   "2024-08-04" \
  --anchor first \
  --window_T 2 --window_H 64 --window_W 64 \
  --stride_T 1 --stride_H 64 --stride_W 64 \
  --max_samples 4
