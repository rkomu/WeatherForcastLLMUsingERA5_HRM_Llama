# MLflow Integration for HRM Vision2Text Training

This document describes how to use MLflow tracking with the HRM Vision2Text training script (`train_hrm_vision2text.py`).

## Installation

MLflow is already included in the requirements.txt file. If you need to install it separately:

```bash
pip install mlflow
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Training with MLflow (Local Tracking)

Run training with MLflow tracking enabled:

```bash
python vision_text/train_hrm_vision2text.py \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 5 \
    --batch_size 2 \
    --mae_ckpt "checkpoints/satswinmae_epoch10.pt" \
    --caption_csv "dataset/weather/weather_august_2024.csv" \
    --mlflow_experiment_name "hrm_vision2text_experiment" \
    --mlflow_run_name "baseline_v1"
```

### 2. Training without MLflow

Disable MLflow tracking:

```bash
python vision_text/train_hrm_vision2text.py \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 5 \
    --batch_size 2 \
    --mae_ckpt "checkpoints/satswinmae_epoch10.pt" \
    --caption_csv "dataset/weather/weather_august_2024.csv" \
    --disable_mlflow
```

### 3. Training with Remote MLflow Server

Connect to a remote MLflow tracking server:

```bash
python vision_text/train_hrm_vision2text.py \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 5 \
    --batch_size 2 \
    --mae_ckpt "checkpoints/satswinmae_epoch10.pt" \
    --caption_csv "dataset/weather/weather_august_2024.csv" \
    --mlflow_tracking_uri "http://mlflow-server:5000" \
    --mlflow_experiment_name "remote_hrm_vision2text"
```

## MLflow Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mlflow_tracking_uri` | MLflow tracking server URI | None (local file store) |
| `--mlflow_experiment_name` | Experiment name | "hrm_vision2text" |
| `--mlflow_run_name` | Run name (auto-generated if None) | None |
| `--mlflow_tags` | Custom tags in format 'key=value' | None |
| `--disable_mlflow` | Disable MLflow logging | False |
| `--log_model_every_n_epochs` | Log model every N epochs (0=only final) | 0 |

## Adding Custom Tags

Add custom tags to organize your experiments:

```bash
python vision_text/train_hrm_vision2text.py \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 5 \
    --batch_size 2 \
    --mae_ckpt "checkpoints/satswinmae_epoch10.pt" \
    --caption_csv "dataset/weather/weather_august_2024.csv" \
    --mlflow_tags env=production model_version=v1.0 dataset=era5_2024
```

## Periodic Model Logging

Log adapter checkpoints every N epochs:

```bash
python vision_text/train_hrm_vision2text.py \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp \
    --epochs 10 \
    --batch_size 2 \
    --mae_ckpt "checkpoints/satswinmae_epoch10.pt" \
    --caption_csv "dataset/weather/weather_august_2024.csv" \
    --log_model_every_n_epochs 3  # Log model every 3 epochs
```

## What Gets Logged

### Parameters
- **Training hyperparameters**: learning rate, batch size, epochs, etc.
- **Model architecture parameters**: embed_dim, depths, heads, n_latents, adapter layers/heads, etc.
- **Data parameters**: variables, window sizes, stride, time ranges, etc.
- **Dataset information**: file counts, sample counts, channel information
- **Model parameters**: adapter, HRM, and MAE parameter counts
- **Vision adapter specific**: d_vis, d_model, n_latents

### Metrics (per epoch)
- Training loss (average per batch)

### Artifacts
- Adapter model checkpoints (PyTorch state dicts)
- Final trained adapter model (MLflow format)

### Tags
- model_type: "HRM_Vision2Text"
- framework: "PyTorch" 
- task: "vision-to-text"
- architecture: "MAE+HRM+VisionAdapter"
- Custom tags specified via `--mlflow_tags`

## Viewing Results

### Local MLflow UI

1. Start the MLflow UI server:
```bash
python start_mlflow_server.py
```
Or directly:
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
python vision_text/train_hrm_vision2text.py \
    --files "dataset/raw_data/nc_*/*.nc" \
    --variables t2m sp ssrd \
    --window_T 8 --window_H 64 --window_W 64 \
    --stride_T 8 --stride_H 64 --stride_W 64 \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --mae_ckpt "checkpoints/satswinmae_epoch10.pt" \
    --embed_dim 96 \
    --n_latents 32 \
    --adapter_layers 2 \
    --adapter_heads 8 \
    --caption_csv "dataset/weather/weather_august_2024.csv" \
    --drop_if_no_caption \
    --mlflow_experiment_name "hrm_vision2text_production" \
    --mlflow_run_name "adapter_large_v1" \
    --mlflow_tags dataset=era5_2024 gpu=a100 architecture=large \
    --log_model_every_n_epochs 5
```

This will:
- Create/use experiment "hrm_vision2text_production"
- Name the run "adapter_large_v1"
- Add custom tags for dataset, GPU type, and architecture
- Log the adapter model every 5 epochs
- Track all metrics and hyperparameters
- Save the final adapter model in MLflow format

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
- Use MLflow Model Registry for versioning important adapter models

### Memory Issues
- Reduce batch size if running out of memory
- Consider using gradient accumulation for larger effective batch sizes
- Monitor GPU memory usage during training

## Integration with Existing Models

The MLflow integration tracks the adapter training specifically. The pre-trained MAE and HRM models are frozen during training, so only the adapter weights are tracked and saved. This allows for:

- Efficient storage (only adapter weights saved)
- Easy deployment (load frozen MAE/HRM + trained adapter)
- Experiment tracking focused on adapter performance
- Comparison of different adapter architectures and hyperparameters
