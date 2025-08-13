# sat_swin_mae/train_mae.py
import os
import argparse
import glob
import random
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .model import SatSwinMAE
from .dataset_era5 import ERA5CubeDataset


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
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--window_T", type=int, default=24)
    ap.add_argument("--window_H", type=int, default=64)
    ap.add_argument("--window_W", type=int, default=64)
    ap.add_argument("--stride_T", type=int, default=24)
    ap.add_argument("--stride_H", type=int, default=32)
    ap.add_argument("--stride_W", type=int, default=32)

    # Training
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
    return ap.parse_args()


def make_loader(files, variables, window, stride, batch_size, shuffle, time_start=None, time_end=None):
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
    print(f"[dataset] T/H/W={ds.T}/{ds.H}/{ds.W}, C={ds.C}, windows={n}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


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

    train_loader = make_loader(
        train_files, args.variables, window, stride, args.batch_size, True,
        time_start=args.time_start, time_end=args.time_end
    )
    val_loader = make_loader(
        val_files, args.variables, window, stride, args.batch_size, False,
        time_start=args.time_start, time_end=args.time_end
    )

    in_chans = int(train_loader.dataset.C)
    if hasattr(val_loader.dataset, "C"):
        assert int(val_loader.dataset.C) == in_chans, \
            f"Train C={in_chans} vs Val C={val_loader.dataset.C} mismatch"

    # (optional) print the expanded channel names for sanity
    chan_names = getattr(train_loader.dataset, "chan_names", None)
    if chan_names:
        print(f"[channels] C={in_chans}: {chan_names}")

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

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # adjusts the learning rate following a cosine curve, decreasing it to a minimum value and then restarting
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        total = 0.0
        for batch in pbar:
            data, valid = unpack_and_move(batch, args.device)
            loss, _ = model(data, compute_loss=True, valid_mask=valid)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data, valid = unpack_and_move(batch, args.device)
                loss, _ = model(data, compute_loss=True, valid_mask=valid)
                vtotal += loss.item() * data.size(0)
        # print(f"vtotal: {vtotal} , len(val_loader.dataset): {len(val_loader.dataset)}")
        if len(val_loader.dataset) > 0:
            val_loss = vtotal / len(val_loader.dataset)
        else:
            val_loss = float('nan')
            print("Warning: Validation dataset is empty.")
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"satswinmae_epoch{epoch}.pt"))
        sched.step()


if __name__ == "__main__":
    main()