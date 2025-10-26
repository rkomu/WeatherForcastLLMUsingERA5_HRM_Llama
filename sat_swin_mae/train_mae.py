# sat_swin_mae/train_mae.py
import os
import argparse
import glob
import random
import time
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np

from .model import SatSwinMAE
from .dataset_era5 import ERA5CubeDataset
from .dataset_cached import CachedCubeMemmapDataset


def expand_files(patterns: List[str]) -> List[str]:
    """Expand a mix of file paths and glob patterns into a sorted, deduped list."""
    out = []
    for p in patterns:
        # If user passed a literal path or a quoted glob, expand it
        expanded = glob.glob(p)
        if not expanded:
            # If no glob hit, assume it's an existing file (or will raise later)
            expanded = [p]
        out.extend(expanded)
    # de-dupe and sort for stability
    out = sorted(list(dict.fromkeys(out)))
    return out


def auto_split_files(files: List[str], val_ratio: float, mode: str, seed: int) -> (List[str], List[str]):
    """Split list of files into train/val by files (not by time)."""
    assert 0.0 < val_ratio < 1.0, "--val_ratio must be in (0,1)"
    files = list(files)
    # ディレクトリごとにグループ化
    from collections import defaultdict
    dir_groups = defaultdict(list)
    for f in files:
        dir_path = os.path.dirname(f)
        dir_groups[dir_path].append(f)

    train_files, val_files = [], []
    rnd = random.Random(seed)
    for dir_path, group in dir_groups.items():
        group = list(group)
        if mode == "random":
            rnd.shuffle(group)
        else:
            group = sorted(group)
        n = len(group)
        n_val = max(1, int(round(n * val_ratio)))
        n_train = max(1, n - n_val)
        if n_train + n_val > n:
            n_val = n - n_train
        train_files.extend(group[:n_train])
        val_files.extend(group[n_train:n_train + n_val])
    # 最終的にシャッフルして返す（順序の偏り防止）
    if mode == "random":
        rnd.shuffle(train_files)
        rnd.shuffle(val_files)
    else:
        train_files = sorted(train_files)
        val_files = sorted(val_files)
    return train_files, val_files


def parse_args():
    ap = argparse.ArgumentParser()
    # New: single entry point
    ap.add_argument("--files", nargs="+", required=False,
                    help="One or more paths/globs. Will be auto-split into train/val.")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of files to use for validation when --files is used.")
    ap.add_argument("--split_mode", type=str, default="chronological",
                    choices=["chronological", "random"],
                    help="How to split files when --files is used.")
    ap.add_argument("--seed", type=int, default=42, help="Seed for random split.")

    # Backward-compatible (optional now). If both are provided, they override --files.
    ap.add_argument("--train_files", nargs="+", required=False, help="(Optional) explicit training files")
    ap.add_argument("--val_files", nargs="+", required=False, help="(Optional) explicit validation files")

    # Data windowing
    '''
    A) Dataset window size (uppercase)

    --window_T 8 --window_H 64 --window_W 64

    This is the clip you extract from ERA5 before it goes into the model.
        •	window_T = timesteps (time depth).
        •	window_H/window_W = latitude/longitude grid size in cells.

    Bigger windows = more context, but more memory and tokens after patching. With your values you take 8×64×64 cubes.

    ⸻

    B) Dataset sliding stride (uppercase)

    --stride_T 4 --stride_H 32 --stride_W 32

    How far you move the crop each time.
        •	Here you use 50% overlap (half strides) in all three dims.
        •	Temporal: 8→4 (half overlap).
        •	Spatial: 64→32 (half overlap).

    How many windows you get (per file region):
    n = floor((Dim - window) / stride) + 1 per dimension; multiply T×H×W counts.

    Example with global grid 721×1440 (from your logs):
        •	H windows: floor((721-64)/32)+1 = 21
        •	W windows: floor((1440-64)/32)+1 = 44
        •	Spatial per timestep: 21×44 = 924 (matches your earlier printout)
    '''
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--window_T", type=int, default=24, help = "This is the clip you extract from ERA5 before it goes into the model. timesteps (time depth)")
    ap.add_argument("--window_H", type=int, default=64, help = "This is the clip you extract from ERA5 before it goes into the model. latitude grid size in cells")
    ap.add_argument("--window_W", type=int, default=64, help = "This is the clip you extract from ERA5 before it goes into the model. longitude grid size in cells.")
    ap.add_argument("--stride_T", type=int, default=24)
    ap.add_argument("--stride_H", type=int, default=32)
    ap.add_argument("--stride_W", type=int, default=32)

    # Training
    '''
    C) Patch embedding (tokenization to patches)

    --patch_t 2 --patch_h 4 --patch_w 4

    SatSwinMAE starts with a Conv3D that chops the cube into non-overlapping 3-D patches (kernel=stride=patch).
        •	Tokens grid (before masking):
        ```
        T' = floor(window_T / patch_t)   = floor(8 / 2)  = 4
        H' = floor(window_H / patch_h)   = floor(64 / 4) = 16
        W' = floor(window_W / patch_w)   = floor(64 / 4) = 16
        num_tokens = T'·H'·W' = 4·16·16 = 1,024
        ```
        	•	Tip: pick window_* as multiples of patch_* so you don’t drop edge cells.

        Larger patches ↓tokens (cheaper but coarser); smaller patches ↑tokens (finer but heavier).

        ⸻

        D) Swin attention window (lowercase)

        --window_t 2 --window_h 8 --window_w 8

        This is the Swin Transformer’s local attention window size in token units, not raw grid cells. It operates after patching, on the T'×H'×W' token grid.

        With the values above:
            •	Token grid: T'×H'×W' = 4×16×16
            •	Swin window: 2×8×8 tokens
            •	Windows per block:
             ```
             (T'/window_t) × (H'/window_h) × (W'/window_w) = (4/2) × (16/8) × (16/8) = 2×2×2 = 8 windows
             ```
        	•	Each attention window sees 2·8·8 = 128 tokens.
         
        Compute per head scales like: #windows × (window_tokens)^2 = 8 × 128^2, which is ~8× cheaper than global attention on 1,024 tokens.

        Rule of thumb: choose window_t/h/w that divide T'/H'/W' cleanly. If not, Swin pads internally.
    '''
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--embed_dim", type=int, default=96)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 2])
    ap.add_argument("--heads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--window_t", type=int, default=2)
    ap.add_argument("--window_h", type=int, default=8)
    ap.add_argument("--window_w", type=int, default=8)
    ap.add_argument("--patch_t", type=int, default=2)
    ap.add_argument("--patch_h", type=int, default=4)
    ap.add_argument("--patch_w", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--time_start", type=str, default=None,
                    help="Start of time range (e.g., '2000-01-01' or '2000-01-01 06:00'). Inclusive.")
    ap.add_argument("--time_end",   type=str, default=None,
                    help="End of time range (e.g., '2000-08-31'). Inclusive.")
    ap.add_argument("--loader_workers", type=int, default=2,
                    help="Number of DataLoader worker processes. Set to 0 to debug single-process loading.")
    ap.add_argument("--loader_prefetch_factor", type=int, default=2,
                    help="prefetch_factor for each worker. Ignored if workers=0.")
    ap.add_argument("--loader_pin_memory", dest="loader_pin_memory", action="store_true",
                    help="Pin CPU tensors before moving to GPU (default).")
    ap.add_argument("--no_loader_pin_memory", dest="loader_pin_memory", action="store_false",
                    help="Disable pin_memory to reduce host RAM pressure.")
    ap.set_defaults(loader_pin_memory=True)
    ap.add_argument("--loader_persistent_workers", dest="loader_persistent_workers", action="store_true",
                    help="Keep DataLoader workers alive between epochs (default when workers>0).")
    ap.add_argument("--no_loader_persistent_workers", dest="loader_persistent_workers", action="store_false",
                    help="Respawn workers every epoch (slower but releases RAM).")
    ap.set_defaults(loader_persistent_workers=True)
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Number of steps to accumulate gradients before each optimizer step.")
    ap.add_argument("--use_amp", action="store_true",
                    help="Enable torch.cuda.amp autocast/GradScaler for mixed precision.")
    ap.add_argument("--cache_train_dir", type=str, default=None,
                    help="Path to memmap cache (from tools/cache_era5_cubes.py) for training set.")
    ap.add_argument("--cache_val_dir", type=str, default=None,
                    help="Path to memmap cache for validation set.")
    
    # MLflow tracking arguments
    ap.add_argument("--mlflow_tracking_uri", type=str, default=None,
                    help="MLflow tracking server URI (e.g., 'http://localhost:5000'). If None, uses local file store.")
    ap.add_argument("--mlflow_experiment_name", type=str, default="sat_swin_mae",
                    help="MLflow experiment name.")
    ap.add_argument("--mlflow_run_name", type=str, default=None,
                    help="MLflow run name. If None, auto-generated.")
    ap.add_argument("--mlflow_tags", nargs="+", default=None,
                    help="MLflow tags in format 'key=value'. Example: --mlflow_tags env=dev model=swinmae")
    ap.add_argument("--disable_mlflow", action="store_true",
                    help="Disable MLflow logging.")
    ap.add_argument("--log_model_every_n_epochs", type=int, default=0,
                    help="Log model checkpoint to MLflow every N epochs. 0 means only log final model.")
    
    return ap.parse_args()


def _default_worker_init(worker_id):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        import numpy as _np
        if hasattr(_np, "seterr"):
            _np.seterr(all="ignore")
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
            if cpus:
                target = cpus[worker_id % len(cpus)]
                os.sched_setaffinity(0, {target})
        except Exception:
            pass


def make_loader(files, variables, window, stride, batch_size, shuffle,
                time_start=None, time_end=None, num_workers=2,
                pin_memory=True, prefetch_factor=2,
                persistent_workers=True, worker_init_fn=_default_worker_init):
    ds = ERA5CubeDataset(
        files, variables, window, stride,
        time_start=time_start, time_end=time_end,
    )
    n = len(ds)
    if n <= 0:
        raise SystemExit(
            f"[train_mae] Dataset produced 0 samples. "
            f"Shapes T/H/W={ds.T}/{ds.H}/{ds.W}, window={window}, stride={stride}, "
            f"time_start={time_start}, time_end={time_end}"
        )
    approx_sample_bytes = ds.C * window["T"] * window["H"] * window["W"] * 4
    approx_batch_gb = (approx_sample_bytes * batch_size) / (1024 ** 3)
    prefetch = prefetch_factor if (num_workers > 0 and prefetch_factor is not None) else 0
    print(
        f"[dataset] T/H/W={ds.T}/{ds.H}/{ds.W}, C={ds.C}, windows={n} | "
        f"~{approx_sample_bytes/1e6:.1f} MB/sample, ~{approx_batch_gb:.2f} GB/batch "
        f"(workers={num_workers}, prefetch={prefetch}, pin_memory={pin_memory})"
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["worker_init_fn"] = worker_init_fn
    return DataLoader(ds, **loader_kwargs)


def make_cached_loader(cache_dir, batch_size, shuffle, num_workers, pin_memory,
                       prefetch_factor, persistent_workers):
    ds = CachedCubeMemmapDataset(cache_dir)
    approx_sample_bytes = np.prod(ds.cube_shape) * 4
    approx_batch_gb = (approx_sample_bytes * batch_size) / (1024 ** 3)
    print(
        f"[cached dataset] dir={cache_dir} samples={len(ds)} shape={ds.cube_shape} | "
        f"~{approx_sample_bytes/1e6:.1f} MB/sample, ~{approx_batch_gb:.2f} GB/batch "
        f"(workers={num_workers}, prefetch={prefetch_factor or 0}, pin_memory={pin_memory})"
    )
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["worker_init_fn"] = _default_worker_init
    return DataLoader(ds, **loader_kwargs)


def unpack_and_move(batch, device):
    """
    Supports: Tensor, (Tensor,), (Tensor, valid_mask), dicts.
    Returns (data_tensor, valid_mask_or_None)
    """
    import torch
    if torch.is_tensor(batch):
        return batch.to(device), None
    if isinstance(batch, (list, tuple)):
        data = batch[0].to(device)
        valid = batch[1].to(device) if len(batch) > 1 and batch[1] is not None else None
        return data, valid
    if isinstance(batch, dict):
        # expect keys like {"data": Tensor, "valid": Tensor?}
        data = batch.get("data").to(device)
        valid = batch.get("valid")
        valid = valid.to(device) if valid is not None else None
        return data, valid
    raise TypeError(f"Unsupported batch type: {type(batch)}")

def main():
    args = parse_args()

    # Initialize MLflow
    if not args.disable_mlflow:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        
        mlflow.set_experiment(args.mlflow_experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=args.mlflow_run_name):
            # Set tags
            mlflow.set_tag("model_type", "SatSwinMAE")
            mlflow.set_tag("framework", "PyTorch")
            mlflow.set_tag("task", "self-supervised learning")
            
            # Add custom tags if provided
            if args.mlflow_tags:
                for tag in args.mlflow_tags:
                    if "=" in tag:
                        key, value = tag.split("=", 1)
                        mlflow.set_tag(key.strip(), value.strip())
                    else:
                        print(f"Warning: Invalid tag format '{tag}', expected 'key=value'")
            
            # Log hyperparameters
            mlflow.log_params({
                "variables": args.variables,
                "window_T": args.window_T,
                "window_H": args.window_H,
                "window_W": args.window_W,
                "stride_T": args.stride_T,
                "stride_H": args.stride_H,
                "stride_W": args.stride_W,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "mask_ratio": args.mask_ratio,
                "embed_dim": args.embed_dim,
                "depths": args.depths,
                "heads": args.heads,
                "window_t": args.window_t,
                "window_h": args.window_h,
                "window_w": args.window_w,
                "patch_t": args.patch_t,
                "patch_h": args.patch_h,
                "patch_w": args.patch_w,
                "device": args.device,
                "val_ratio": args.val_ratio,
                "split_mode": args.split_mode,
                "seed": args.seed,
                "time_start": args.time_start,
                "time_end": args.time_end,
                "log_model_every_n_epochs": args.log_model_every_n_epochs,
                "loader_workers": args.loader_workers,
                "loader_prefetch_factor": args.loader_prefetch_factor,
                "loader_pin_memory": args.loader_pin_memory,
                "loader_persistent_workers": args.loader_persistent_workers,
                "grad_accum_steps": args.grad_accum_steps,
                "use_amp": args.use_amp,
                "cache_train_dir": args.cache_train_dir,
                "cache_val_dir": args.cache_val_dir,
            })
            
            run_training(args)
    else:
        run_training(args)


def run_training(args):
    """Main training logic separated for MLflow integration."""
    # Decide file splits
    if args.train_files and args.val_files:
        train_files = expand_files(args.train_files)
        val_files = expand_files(args.val_files)
    else:
        if not args.files:
            raise SystemExit("Please pass either --files (for auto split) or both --train_files and --val_files.")
        all_files = expand_files(args.files)
        if len(all_files) < 2:
            raise SystemExit(f"Need at least 2 files for auto split, got {len(all_files)}")
        train_files, val_files = auto_split_files(
            all_files, val_ratio=args.val_ratio, mode=args.split_mode, seed=args.seed
        )

    print(f"[split] train_files={len(train_files)}  val_files={len(val_files)}")
    if len(train_files) < 1 or len(val_files) < 1:
        raise SystemExit("Split produced empty train/val sets; adjust --val_ratio or provide more files.")

    window = {"T": args.window_T, "H": args.window_H, "W": args.window_W}
    stride = {"T": args.stride_T, "H": args.stride_H, "W": args.stride_W}

    loader_opts = dict(
        num_workers=args.loader_workers,
        pin_memory=args.loader_pin_memory,
        prefetch_factor=args.loader_prefetch_factor,
        persistent_workers=args.loader_persistent_workers if args.loader_workers > 0 else False,
    )

    use_cache = args.cache_train_dir or args.cache_val_dir
    if use_cache:
        if not (args.cache_train_dir and args.cache_val_dir):
            raise SystemExit("Please provide both --cache_train_dir and --cache_val_dir when using cached data.")
        train_loader = make_cached_loader(
            args.cache_train_dir, args.batch_size, True,
            **loader_opts
        )
        val_loader = make_cached_loader(
            args.cache_val_dir, args.batch_size, False,
            **loader_opts
        )
        cache_window = getattr(train_loader.dataset, "window", None)
        if isinstance(cache_window, dict):
            window = cache_window
        cache_stride = getattr(train_loader.dataset, "stride", None)
        if isinstance(cache_stride, dict):
            stride = cache_stride
    else:
        train_loader = make_loader(
            train_files, args.variables, window, stride, args.batch_size, True,
            time_start=args.time_start, time_end=args.time_end, **loader_opts
        )
        val_loader = make_loader(
            val_files, args.variables, window, stride, args.batch_size, False,
            time_start=args.time_start, time_end=args.time_end, **loader_opts
        )

    in_chans = int(train_loader.dataset.C)
    if hasattr(val_loader.dataset, "C"):
        assert int(val_loader.dataset.C) == in_chans, \
            f"Train C={in_chans} vs Val C={val_loader.dataset.C} mismatch"

    # (optional) print the expanded channel names for sanity
    chan_names = getattr(train_loader.dataset, "chan_names", None)
    if chan_names:
        print(f"[channels] C={in_chans}: {chan_names}")

    # Log additional dataset information to MLflow
    if not args.disable_mlflow:
        mlflow.log_params({
            "train_files_count": len(train_files),
            "val_files_count": len(val_files),
            "in_chans": in_chans,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        })
        if chan_names:
            mlflow.log_param("channel_names", str(chan_names))

    # model
    model = SatSwinMAE(
        in_chans=in_chans,
        out_chans=in_chans,   # reconstruct all channels
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.heads),
        window_size=(args.window_t, args.window_h, args.window_w),
        patch_size=(args.patch_t, args.patch_h, args.patch_w),
        mask_ratio=args.mask_ratio
    ).to(args.device)

    # Log model info
    if not args.disable_mlflow:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    amp_enabled = bool(args.use_amp and ("cuda" in str(args.device).lower()))
    if args.use_amp and not amp_enabled:
        print("[train_mae] --use_amp requested but CUDA device not detected; running in full precision.")
    scaler = GradScaler(enabled=amp_enabled)

    # adjusts the learning rate following a cosine curve, decreasing it to a minimum value and then restarting
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(args.out_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        train_loss_sum = 0.0
        train_batch_count = 0
        first_batch_t0 = time.perf_counter()
        first_batch_logged = False
        accum_steps = max(1, args.grad_accum_steps)
        opt.zero_grad()
        num_batches = len(train_loader)
        prev_step_end = time.perf_counter()

        for step, batch in enumerate(pbar, start=1):
            step_start = time.perf_counter()
            loader_wait = step_start - prev_step_end
            if not first_batch_logged:
                latency = time.perf_counter() - first_batch_t0
                pbar.write(f"[loader] First batch ready after {latency:.1f}s "
                           f"(batch_size={args.batch_size}, num_workers={train_loader.num_workers})")
                first_batch_logged = True
            data, valid = unpack_and_move(batch, args.device)
            with autocast(enabled=amp_enabled):
                loss, _ = model(data, compute_loss=True, valid_mask=valid)
            loss_value = loss.item()
            pbar.set_postfix(loss=f"{loss_value:.4f}")

            scaled_loss = loss / accum_steps
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if step % accum_steps == 0 or step == num_batches:
                if amp_enabled:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if amp_enabled:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
            
            train_loss_sum += loss_value
            train_batch_count += 1
            global_step += 1
            step_end = time.perf_counter()
            compute_time = step_end - step_start
            step_time = max(step_end - prev_step_end, 1e-9)
            gpu_active_ratio = compute_time / step_time
            prev_step_end = step_end
            if not args.disable_mlflow:
                mlflow.log_metrics(
                    {
                        "train_step_loss": loss_value,
                        "train_step_lr": opt.param_groups[0]["lr"],
                        "loader_wait_s": loader_wait,
                        "compute_s": compute_time,
                        "gpu_active_ratio": gpu_active_ratio,
                    },
                    step=global_step,
                )

        # Calculate average training loss
        avg_train_loss = train_loss_sum / train_batch_count if train_batch_count > 0 else float('nan')

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data, valid = unpack_and_move(batch, args.device)
                with autocast(enabled=amp_enabled):
                    loss, _ = model(data, compute_loss=True, valid_mask=valid)
                vtotal += loss.item() * data.size(0)
        # print(f"vtotal: {vtotal} , len(val_loader.dataset): {len(val_loader.dataset)}")
        if len(val_loader.dataset) > 0:
            val_loss = vtotal / len(val_loader.dataset)
        else:
            val_loss = float('nan')
            print("Warning: Validation dataset is empty.")
            
        # Log metrics to MLflow
        if not args.disable_mlflow:
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": sched.get_last_lr()[0],
            }, step=epoch)
            
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.out_dir, f"satswinmae_epoch{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Log model artifact to MLflow
        if not args.disable_mlflow:
            # Log checkpoint artifact for every specified epoch or final epoch
            should_log_model = (
                epoch == args.epochs or  # Always log final model
                (args.log_model_every_n_epochs > 0 and epoch % args.log_model_every_n_epochs == 0)
            )
            
            if should_log_model:
                try:
                    # Log the PyTorch model
                    mlflow.pytorch.log_model(
                        model, 
                        f"model_epoch_{epoch}",
                        registered_model_name=f"{args.mlflow_experiment_name}_model" if epoch == args.epochs else None
                    )
                    # Log the checkpoint file
                    mlflow.log_artifact(checkpoint_path, "checkpoints")
                    print(f"Logged model and checkpoint for epoch {epoch}")
                except Exception as e:
                    print(f"Warning: Failed to log model to MLflow: {e}")
            
        sched.step()


if __name__ == "__main__":
    main()
