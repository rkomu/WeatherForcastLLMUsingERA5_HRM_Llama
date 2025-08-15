#!/bin/bash

python -m vision_text.infer_hrm_vision2text \
  --mae_ckpt checkpoints/satswinmae_epoch10.pt \
  --adapter_ckpt checkpoints_v2t/adapter_epoch20.pt \
  --files ./dataset/raw_data/**/*.nc \
  --variables u10 v10 r sp ssrd t cp \
  --date 2025-08-01 \
  --anchor last \
  --window_T 8 --window_H 64 --window_W 64 \
  --stride_T 8 --stride_H 64 --stride_W 64 \
  --max_samples 4
