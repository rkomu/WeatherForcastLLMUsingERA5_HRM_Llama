# Repository Guidelines

# WeatherLM Agents

This guide describes how to operate the WeatherLM vision-to-text agents when you **only use the TinyLlama language model** alongside the frozen SatSwinMAE vision encoder. It consolidates the architecture context from `README.md` and the experiment-tracking workflows from `MLFLOW_GUIDE.md`.

## Agent Stack Overview
- **Vision encoder**: SatSwinMAE remains frozen and turns ERA5 cubes (`channels × time × lat × lon`) into latent tokens. Keep window dimensions aligned with the encoder’s patch sizes.
- **Vision→text adapter**: A lightweight MLP maps the SatSwinMAE latents (`Dv`) into TinyLlama’s embedding space (`dH`). Only this adapter (and optional LoRA layers) train.
- **Language agent (TinyLlama-1.1B)**: Consumes soft prompt tokens emitted by the adapter and generates weather narratives. HRM is not part of this workflow.

## TinyLlama Training Agent

### Prepare data
- Curate ERA5 NetCDF paths for the variables that match the SatSwinMAE checkpoint’s `in_chans`.
- Assemble caption CSVs with at least `date`, `location`, and `event description` (rename columns via CLI flags if needed).
- Choose window (`--window_T/H/W`) and stride (`--stride_T/H/W`) values so that the dataset can tile the ERA5 grids without dropping context.

### Launch training
```bash
./train_visiontext_tinylama.sh
```

Key switches:
- `--use_qlora` plus LoRA hyperparameters enables memory-efficient fine-tuning when VRAM is tight.
- `--drop_if_no_caption` filters unmatched ERA5 windows to keep the supervision clean.
- `--anchor` (`first|middle|last`) controls which timestep is aligned with each caption.

### Track experiments with MLflow
The TinyLlama trainer accepts the same MLflow flags documented for SatSwinMAE:
```bash
python -m vision_text.train_tinyllama_vision2text \
  ... \
  --mlflow_experiment_name "tinyllama_weatherlm" \
  --mlflow_run_name "tokyo_windows_v1" \
  --mlflow_tags env=dev backbone=satswinmae adapter=mlp32
```
- Use `--mlflow_tracking_uri` to target a remote server or leave unset for the default `mlruns/` folder.
- Disable logging with `--disable_mlflow` if you do not want tracking artifacts.
- Combine with `--log_model_every_n_epochs N` to archive adapter checkpoints periodically.

## TinyLlama Inference Agent
Run inference once training converges:
```bash
python infer_tinyllama_vision2text.py \
  --files "dataset/raw_data/nc_*/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --date 2024-03-15 \
  --mae_ckpt checkpoints/satswinmae_epoch59.pt \
  --adapter_ckpt checkpoints_v2t/tinyllama_adapter_final.pt \
  --max_new_tokens 128 --temperature 0.7
```
- Adjust `--max_new_tokens` and `--temperature` for longer or more diverse narratives.
- Ensure the inference variables and windows match what the adapter saw during training.

## Monitoring Runs
- Start the local UI with `mlflow ui` and open http://localhost:5000 to inspect TinyLlama training runs.
- For remote tracking servers, point `--mlflow_tracking_uri` to the deployed endpoint and browse there.
- MLflow captures hyperparameters, per-epoch losses, and saved adapter checkpoints to streamline comparisons across experiments.

## Troubleshooting & Tips
- Verify SatSwinMAE input channels and ERA5 variables line up; channel mismatches prevent forward passes.
- Flat loss around 10–11 usually indicates the adapter parameters are not included in the optimizer or the soft prompts were detached.
- Keep TinyLlama’s max sequence length ≥ `n_latents + max_caption_tokens`.
- Start in fp32 for stability, then switch to bf16 once gradients look healthy.

## SatSwinMAE Performance Notes
- **Loader parallelism**: `train_mae.py` now exposes `--loader_workers`, `--loader_prefetch_factor`, and `--loader_pin_memory` so you can saturate an RTX 3090 once the cubes are light enough. The default `train_mae.sh` sets `--loader_workers 6 --loader_prefetch_factor 4`; lower these on slower disks.
- **Gradient accumulation**: Use `--grad_accum_steps` to keep an effective large batch while only materializing a handful of windows per step. The launcher currently pairs `--batch_size 4` with `--grad_accum_steps 4` to emulate a batch of 16.
- **Cache ERA5 cubes**: For sustained throughput, run an offline job that walks `dataset/raw_data/**/*.nc`, extracts `(C,T,H,W)` windows once, and stores them as `.npy`/Zarr chunks on NVMe. Point future training runs at the cached directory by swapping in a dataset wrapper that memory-maps the cubes so PyTorch never re-reads NetCDF or recomputes normalization.
- **Mixed precision (AMP)**: Pass `--use_amp` to `train_mae.py` once data loading is no longer the bottleneck. The loop now wraps forward/backward passes with `torch.cuda.amp.autocast` and `GradScaler` to cut GPU memory and speed math on Ada/Ampere GPUs.



## Project Structure & Module Organization
- `sat_swin_mae/` holds the masked autoencoder, ERA5 tiling utilities, and `train_mae.py` entry point; adjust configs in `config/cfg_pretrain.yaml` and `config/arch/` before pretraining.
- `vision_text/` contains the adapter, HRM/TinyLlama training loops, and dataset bridge classes; related shell runners live at the repo root (`train_visiontexthrm.sh`, `train_visiontext_tinylama.sh`).
- `dataset/` manages raw ERA5 pulls and caption CSV preparation, while `tools/` provides ancillary scripts such as `build_hrm_dataset.py`.
- Checkpoints live in `checkpoints*/`; diagrams and static assets sit in `diagrams/` and `assets/`.

## Build, Test, and Development Commands
- Create the base environment with `conda env create -f environment.yml` followed by `pip install -r requirements.txt` if you need extras.
- Pretrain SatSwinMAE locally: `python -m sat_swin_mae.train_mae --files "dataset/raw_data/**/*.nc" --config config/cfg_pretrain.yaml`.
- Fine-tune the vision-to-text adapter using `bash train_visiontexthrm.sh` or the TinyLlama variant; edit the scripts rather than inlining long CLI invocations.
- For evaluation, run `python evaluate.py checkpoint=checkpoints/satswinmae_epochXX.pt` to regenerate metrics and logs.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation; keep line length ≤ 100 to match existing modules such as `vision_text/train_hrm_vision2text.py`.
- Prefer expressive snake_case for Python symbols and CamelCase for classes; mirror existing variable names (`window_T`, `mask_ratio`) when extending dataclasses or CLI arguments.
- Include type hints and docstrings when touching public APIs; keep logging consistent with the structured `mlflow` logging already in place.

## Testing Guidelines
- There is no standalone pytest suite; lean on script-level evaluation. Always run the relevant training script with `--eval_every` enabled and review MLflow artifacts before opening a PR.
- Use `python evaluate.py checkpoint=...` after any MAE changes and spot-check generated captions via `--eval_log_samples` for adapter updates.
- Record manual verification steps in the PR description (e.g., “train_visiontexthrm.sh for 2 epochs on sample ERA5 split, loss < 3.5”).

## Commit & Pull Request Guidelines
- Git history favors descriptive, sentence-style commits (`Enhance logging and progress tracking...`); keep messages imperative and scoped.
- Reference issues or PR numbers using `(#123)` when applicable and avoid bundling unrelated changes.
- PRs should explain the experiment context, note required checkpoints or data slices, and attach qualitative samples or MLflow links when model behavior changes.

## Configuration & Tracking
- Store reusable hyperparameters under `config/`; update YAMLs in tandem with code so `all_config.yaml` snapshots remain trustworthy.
- If you modify tracking behavior, ensure `MLFLOW_GUIDE.md` reflects the new workflow and keep `start_mlflow_server.py` defaults in sync with README instructions.
