#!/bin/bash

export TOKENIZERS_PARALLELISM=false

# Ensure conda is available and create/activate environment if needed
ENV_NAME="WeatherForcastWithHrmAndSatSwinMAE"
ENV_FILE="environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Please install Miniconda or Anaconda and re-run this script."
  exit 1
fi

# Make conda commands available in non-login shells
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"

# Create the environment from environment.yml if it does not exist
if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null 2>&1; then
  echo "Activating existing conda env: $ENV_NAME"
else
  if [ -f "$ENV_FILE" ]; then
    echo "Creating conda env '$ENV_NAME' from $ENV_FILE..."
    conda env create -f "$ENV_FILE" -n "$ENV_NAME" || { echo "Failed to create environment from $ENV_FILE"; exit 1; }
  else
    echo "No $ENV_FILE found. Creating minimal env '$ENV_NAME' with python..."
    conda create -y -n "$ENV_NAME" python=3.10 || { echo "Failed to create minimal environment"; exit 1; }
  fi
fi

conda activate "$ENV_NAME"

PYTHON_BIN="$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m vision_text.train_tinyllama_vision2text \
  --files "dataset/raw_data/nc_*/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --window_T 48 --window_H 64 --window_W 64 \
  --stride_T 24 --stride_H 32 --stride_W 32 \
  --time_start 2024-01-01 --time_end 2024-01-15 \
  \
  --caption_csv dataset/weather/tokyo_weather_2023-2025.csv \
  --caption_date_col date --caption_text_col "event description" \
  --caption_location_col location --caption_location_values Tokyo \
  --drop_if_no_caption --anchor last \
  \
  --mae_ckpt checkpoints/satswinmae_epoch59.pt \
  --model_name TinyLlama/TinyLlama_v1.1 \
  --batch_size 32 --epochs 10 --lr 0.0001 \
  --n_latents 32 --adapter_layers 2 --adapter_heads 8 \
  \
  --eval_every 2 --gen_samples 3 --gen_max_new_tokens 128 \
  \
  --mlflow_experiment_name tinyllama_vision2text \
  --mlflow_run_name aug2024_adapter \
  --split_mode random \
  --use_qlora --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
