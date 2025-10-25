#!/bin/bash

ENV_NAME="WeatherForcastWithHrmAndSatSwinMAE"
ENV_FILE="environment.yml"
export TOKENIZERS_PARALLELISM=false

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

"$PYTHON_BIN" -m sat_swin_mae.train_mae \
  --files "./dataset/raw_data/**/*.nc" \
  --batch_size 4 --epochs 5 \
  --grad_accum_steps 4 \
  --val_ratio 0.3 --split_mode random --seed 123 \
  --window_T 168 --window_H 64 --window_W 64 \
  --stride_T 168 --stride_H 32 --stride_W 32 \
  --mask_ratio 0.75 \
  --window_t 2 --window_h 8 --window_w 8 \
  --patch_t 2 --patch_h 4 --patch_w 4 \
  --variables u10 v10 r sp ssrd t cp \
  --time_start "2024-01-01" \
  --time_end   "2024-08-31" \
  --loader_workers 6 \
  --loader_prefetch_factor 4 \
  --mlflow_experiment_name "mae_training" \
  --mlflow_run_name "swinmae_v2.1" \
  --mlflow_tags dataset=era5_2024 gpu=3090 experiment_type=baseline \
  --log_model_every_n_epochs 2
