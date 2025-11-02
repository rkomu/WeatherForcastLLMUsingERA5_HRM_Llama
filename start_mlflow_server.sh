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

"$PYTHON_BIN" start_mlflow_server.py
