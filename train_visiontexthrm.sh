#!/bin/bash

python3 -m vision_text.train_hrm_vision2text \
  --mae_ckpt checkpoints/satswinmae_epoch10.pt \
  --files ./dataset/raw_data/**/*.nc \
  --variables u10 v10 r sp ssrd t cp \
  --caption_csv ./dataset/weather/weather_august_2024.csv \
  --time_start "2024-08-01" --time_end "2024-08-31" \
  --eval_every 1 --eval_max_samples 64 --eval_log_samples 10 \
  --window_T 8 --window_H 64 --window_W 64 \
  --stride_T 8 --stride_H 64 --stride_W 64 \
  --n_latents 32 --epochs 1000 --batch_size 64 --lr 0.01 \
  --max_gen_len 64 --temperature 0.0 --top_k 0 \
  --resume_adapter_ckpt checkpoints_v2t/adapter_epoch57.pt
