# MLflow Integration for SAT SwinMAE Training

This document describes how to use MLflow tracking with the SAT SwinMAE training script.

## Installation

First, install MLflow:

```bash
pip install mlflow
```

Or install all requirements including MLflow:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Training with MLflow (Local Tracking)

Run training with MLflow tracking enabled:

```bash
python -m sat_swin_mae.train_mae \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 10 \
    --batch_size 2 \
    --mlflow_experiment_name "my_experiment" \
    --mlflow_run_name "my_run_v1"
```

### 2. Training without MLflow

Disable MLflow tracking:

```bash
python -m sat_swin_mae.train_mae \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 10 \
    --disable_mlflow
```

### 3. Training with Remote MLflow Server

Connect to a remote MLflow tracking server:

```bash
python -m sat_swin_mae.train_mae \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 10 \
    --mlflow_tracking_uri "http://mlflow-server:5000" \
    --mlflow_experiment_name "remote_experiment"
```

## MLflow Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mlflow_tracking_uri` | MLflow tracking server URI | None (local file store) |
| `--mlflow_experiment_name` | Experiment name | "sat_swin_mae" |
| `--mlflow_run_name` | Run name (auto-generated if None) | None |
| `--mlflow_tags` | Custom tags in format 'key=value' | None |
| `--disable_mlflow` | Disable MLflow logging | False |
| `--log_model_every_n_epochs` | Log model every N epochs (0=only final) | 0 |

## Adding Custom Tags

Add custom tags to organize your experiments:

```bash
python -m sat_swin_mae.train_mae \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 10 \
    --mlflow_tags env=production model_version=v2.1 dataset=era5_2024
```

## Periodic Model Logging

Log model checkpoints every N epochs:

```bash
python -m sat_swin_mae.train_mae \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 20 \
    --log_model_every_n_epochs 5  # Log model every 5 epochs
```

## What Gets Logged

### Parameters
- All training hyperparameters (learning rate, batch size, epochs, etc.)
- Model architecture parameters (embed_dim, depths, heads, etc.)
- Data parameters (variables, window sizes, stride, etc.)
- Dataset information (file counts, sample counts, channel names)
- Model parameters (total and trainable parameter counts)

### Metrics (per epoch)
- Training loss (average per batch)
- Validation loss
- Learning rate

### Artifacts
- Model checkpoints (PyTorch state dicts)
- Final trained model (MLflow format)

### Tags
- model_type: "SatSwinMAE"
- framework: "PyTorch" 
- task: "self-supervised learning"
- Custom tags specified via `--mlflow_tags`

## Viewing Results

### Local MLflow UI

1. Start the MLflow UI server:
```bash
mlflow ui
```

2. Open http://localhost:5000 in your browser

3. Navigate to your experiment to view runs, metrics, and artifacts

### Remote MLflow Server

If using a remote MLflow server, navigate to the server URL to view results.

## Example: Complete Training Run

```bash
# Full example with all MLflow features
python -m sat_swin_mae.train_mae \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp ssrd \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --mask_ratio 0.8 \
    --mlflow_experiment_name "era5_pretraining" \
    --mlflow_run_name "swinmae_large_v1" \
    --mlflow_tags dataset=era5_2020_2024 gpu=a100 batch_size=4 \
    --log_model_every_n_epochs 10
```

This will:
- Create/use experiment "era5_pretraining"
- Name the run "swinmae_large_v1"
- Add custom tags for dataset, GPU type, and batch size
- Log the model every 10 epochs
- Track all metrics and hyperparameters
- Save the final model in MLflow format

## Troubleshooting

### MLflow Not Found
```bash
pip install mlflow
```

### Permission Issues with Local Tracking
Make sure you have write permissions in the current directory where `mlruns/` folder will be created.

### Remote Server Connection Issues
- Verify the tracking URI is correct
- Check network connectivity to the MLflow server
- Ensure the server is running and accessible

### Large Model Storage
- Use `--log_model_every_n_epochs 0` to only log the final model
- Consider using a shared storage backend for large models
- Use MLflow Model Registry for versioning important models
