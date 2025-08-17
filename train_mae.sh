#!/bin/bash

python3 -m sat_swin_mae.train_mae \
  --files "./dataset/raw_data/**/*.nc" \
  --batch_size 128 --epochs 5 \
  --val_ratio 0.3 --split_mode random --seed 123 \
  --window_T 8 --window_H 64 --window_W 64 \
  --stride_T 4 --stride_H 32 --stride_W 32 \
  --mask_ratio 0.75 \
  --window_t 2 --window_h 8 --window_w 8 \
  --patch_t 2 --patch_h 4 --patch_w 4 \
  --variables u10 v10 r sp ssrd t cp \
  --time_start "2024-08-01" \
  --time_end   "2024-08-31" \
  --mlflow_experiment_name "era5_pretraining" \
  --mlflow_run_name "swinmae_v2.1" \
  --mlflow_tags dataset=era5_2024 gpu=3090 experiment_type=baseline \
  --log_model_every_n_epochs 2
