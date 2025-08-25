#!/bin/bash


# python3 -m vision_text.train_tinyllama_vision2text \
#   --files "dataset/raw_data/**/*.nc" \
#   --variables u10 v10 r sp ssrd t cp \
#   --window_T 8 --window_H 64 --window_W 64 \
#   --stride_T 8 --stride_H 64 --stride_W 64 \
#   --epochs 5 --batch_size 2 --lr 1e-4 \
#   --mae_ckpt checkpoints/satswinmae_epoch10.pt \
#   --caption_csv dataset/weather/weather_august_2024.csv \
#   --drop_if_no_caption \
#   --n_latents 32 --adapter_layers 2 --adapter_heads 8

export TOKENIZERS_PARALLELISM=false

python -m vision_text.train_tinyllama_vision2text \
  --files "dataset/raw_data/nc_*/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --window_T 8 --window_H 64 --window_W 64 \
  --stride_T 4 --stride_H 32 --stride_W 32 \
  --time_start 2024-01-01 --time_end 2024-05-31 \
  \
  --caption_csv dataset/weather/weather_jan_may_2024.csv \
  --caption_date_col date --caption_text_col "event description" \
  --drop_if_no_caption --anchor last \
  \
  --mae_ckpt checkpoints/satswinmae_epoch59.pt \
  --model_name TinyLlama/TinyLlama_v1.1 \
  --batch_size 32 --epochs 10 --lr 1e-4 \
  --n_latents 32 --adapter_layers 2 --adapter_heads 8 \
  \
  --eval_every 2 --gen_samples 3 --gen_max_new_tokens 64 \
  \
  --mlflow_experiment_name tinyllama_vision2text \
  --mlflow_run_name aug2024_adapter \
  --split_mode random
  # --mlflow_tags mae_window=2x8x8 vars=7 in_chans=17