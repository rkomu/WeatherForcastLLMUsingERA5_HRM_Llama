#!/bin/bash

python3 -m vision_text.train_hrm_vision2text \
  --files "./dataset/raw_data/**/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --mae_ckpt checkpoints/satswinmae_epoch10.pt \
  --window_T 8 --window_H 64 --window_W 64 \
  --stride_T 8 --stride_H 64 --stride_W 64 \
  --n_latents 32 --epochs 1 --batch_size 4 --lr 1e-4 \
    --caption_csv ./dataset/weather/weather_august_2024.csv