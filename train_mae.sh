#!/bin/bash

ENV_NAME="WeatherForcastWithHrmAndSatSwinMAE"
ENV_FILE="environment.yml"
export TOKENIZERS_PARALLELISM=false

export CACHE_BUILD=1
export CACHE_TRAIN_DIR=/mnt/nvme/cache/train
export CACHE_VAL_DIR=/mnt/nvme/cache/val
export CACHE_TRAIN_FILES="./dataset/raw_data/train/**/*.nc"
export CACHE_VAL_FILES="./dataset/raw_data/val/**/*.nc"

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
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

WINDOW_T=168
WINDOW_H=64
WINDOW_W=64
STRIDE_T=168
STRIDE_H=32
STRIDE_W=32
MASK_RATIO=0.75
WINDOW_t=2
WINDOW_h=8
WINDOW_w=8
PATCH_t=2
PATCH_h=4
PATCH_w=4
TIME_START="2024-01-01"
TIME_END="2024-08-31"
VARIABLES=(u10 v10 r sp ssrd t cp)

read -r -a CACHE_TRAIN_PATTERNS <<< "${CACHE_TRAIN_FILES:-}"
read -r -a CACHE_VAL_PATTERNS <<< "${CACHE_VAL_FILES:-}"

build_cache_if_missing() {
  local label="$1"
  local out_dir="$2"
  shift 2
  local files=("$@")

  if [[ -z "$out_dir" || ${#files[@]} -eq 0 ]]; then
    echo "[cache:$label] Missing output dir or file patterns; skipping cache build."
    return
  fi

  if [[ -f "$out_dir/meta.json" ]]; then
    echo "[cache:$label] Existing cache detected at $out_dir; skipping build."
    return
  fi

  echo "[cache:$label] Building cache at $out_dir"
  if ! mkdir -p "$out_dir"; then
    echo "[cache:$label] ERROR: Cannot create $out_dir (permission denied?)." >&2
    return 1
  fi
  if ! "$PYTHON_BIN" tools/cache_era5_cubes.py \
    --files "${files[@]}" \
    --variables "${VARIABLES[@]}" \
    --window_T "$WINDOW_T" --window_H "$WINDOW_H" --window_W "$WINDOW_W" \
    --stride_T "$STRIDE_T" --stride_H "$STRIDE_H" --stride_W "$STRIDE_W" \
    --time_start "$TIME_START" --time_end "$TIME_END" \
    --output_dir "$out_dir"; then
    echo "[cache:$label] ERROR: Cache build failed." >&2
    return 1
  fi
}

if [[ "${CACHE_BUILD:-0}" != 0 ]]; then
  build_cache_if_missing "train" "${CACHE_TRAIN_DIR:-}" "${CACHE_TRAIN_PATTERNS[@]}" || CACHE_BUILD_FAILED=1
  build_cache_if_missing "val" "${CACHE_VAL_DIR:-}" "${CACHE_VAL_PATTERNS[@]}" || CACHE_BUILD_FAILED=1
fi

CACHE_FLAGS=()
if [[ -n "${CACHE_TRAIN_DIR:-}" && -n "${CACHE_VAL_DIR:-}" && -f "${CACHE_TRAIN_DIR}/meta.json" && -f "${CACHE_VAL_DIR}/meta.json" && "${CACHE_BUILD_FAILED:-0}" != 1 ]]; then
  CACHE_FLAGS=(--cache_train_dir "$CACHE_TRAIN_DIR" --cache_val_dir "$CACHE_VAL_DIR")
else
  if [[ -n "${CACHE_TRAIN_DIR:-}" || -n "${CACHE_VAL_DIR:-}" ]]; then
    echo "[cache] Warning: Cache directories missing meta.json or build failed; falling back to raw NetCDF files."
  fi
fi

"$PYTHON_BIN" -m sat_swin_mae.train_mae \
  --files "./dataset/raw_data/**/*.nc" \
  --batch_size 16 --epochs 5 \
  --grad_accum_steps 3 \
  --val_ratio 0.3 --split_mode random --seed 123 \
  --window_T "$WINDOW_T" --window_H "$WINDOW_H" --window_W "$WINDOW_W" \
  --stride_T "$STRIDE_T" --stride_H "$STRIDE_H" --stride_W "$STRIDE_W" \
  --mask_ratio "$MASK_RATIO" \
  --window_t "$WINDOW_t" --window_h "$WINDOW_h" --window_w "$WINDOW_w" \
  --patch_t "$PATCH_t" --patch_h "$PATCH_h" --patch_w "$PATCH_w" \
  --variables "${VARIABLES[@]}" \
  --time_start "$TIME_START" \
  --time_end   "$TIME_END" \
  --loader_workers 8 \
  --loader_prefetch_factor 6 \
  --loader_persistent_workers \
  --use_amp \
  --mlflow_experiment_name "mae_training" \
  --mlflow_run_name "swinmae_v2.1" \
  --mlflow_tags dataset=era5_2024 gpu=3090 experiment_type=baseline \
  --log_model_every_n_epochs 2 \
  "${CACHE_FLAGS[@]}"
