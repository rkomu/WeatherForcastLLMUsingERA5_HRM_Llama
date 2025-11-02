"""
Offline utility to precompute ERA5 cubes and store them as contiguous memmap
arrays for faster SatSwinMAE training.

Usage example:

python tools/cache_era5_cubes.py \
  --files "dataset/raw_data/**/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --window_T 168 --window_H 64 --window_W 64 \
  --stride_T 168 --stride_H 32 --stride_W 32 \
  --time_start 2024-01-01 --time_end 2024-08-31 \
  --output_dir /mnt/nvme/era5_cache/train
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sat_swin_mae.dataset_era5 import ERA5CubeDataset


def parse_args():
    ap = argparse.ArgumentParser(description="Materialize ERA5 cubes into a memmap cache.")
    ap.add_argument("--files", nargs="+", required=True, help="NetCDF glob(s) to ingest.")
    ap.add_argument("--variables", nargs="+", required=True, help="ERA5 variables to include.")
    ap.add_argument("--window_T", type=int, default=168)
    ap.add_argument("--window_H", type=int, default=64)
    ap.add_argument("--window_W", type=int, default=64)
    ap.add_argument("--stride_T", type=int, default=168)
    ap.add_argument("--stride_H", type=int, default=32)
    ap.add_argument("--stride_W", type=int, default=32)
    ap.add_argument("--time_start", type=str, default=None)
    ap.add_argument("--time_end", type=str, default=None)
    ap.add_argument("--output_dir", type=str, required=True, help="Where to write the memmap files.")
    ap.add_argument("--batch_size", type=int, default=16, help="Loader batch size for dumping.")
    ap.add_argument("--loader_workers", type=int, default=4, help="Workers for the dumping DataLoader.")
    return ap.parse_args()


def main():
    args = parse_args()
    expanded_files = []
    for pattern in args.files:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            expanded_files.extend(sorted(matches))
        else:
            expanded_files.append(pattern)
    if not expanded_files:
        raise SystemExit("No files matched the provided patterns.")
    window = {"T": args.window_T, "H": args.window_H, "W": args.window_W}
    stride = {"T": args.stride_T, "H": args.stride_H, "W": args.stride_W}
    ds = ERA5CubeDataset(
        files=expanded_files,
        variables=args.variables,
        window=window,
        stride=stride,
        time_start=args.time_start,
        time_end=args.time_end,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_samples = len(ds)
    cube_shape = (ds.C, window["T"], window["H"], window["W"])
    valid_shape = (window["T"], window["H"], window["W"])
    cubes = np.memmap(out_dir / "cubes.bin", dtype=np.float32, mode="w+", shape=(num_samples,) + cube_shape)
    valid = np.memmap(out_dir / "valid_mask.bin", dtype=np.bool_, mode="w+", shape=(num_samples,) + valid_shape)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=False,
    )
    offset = 0
    for batch_cubes, batch_valid in tqdm(loader, total=len(loader), desc="Caching cubes"):
        bsz = batch_cubes.size(0)
        cubes[offset:offset + bsz] = batch_cubes.numpy()
        valid[offset:offset + bsz] = batch_valid.numpy()
        offset += bsz
    cubes.flush()
    valid.flush()

    meta = {
        "num_samples": num_samples,
        "cube_shape": list(cube_shape),
        "valid_shape": list(valid_shape),
        "cube_dtype": "float32",
        "valid_dtype": "bool",
        "chan_names": getattr(ds, "chan_names", None),
        "window": window,
        "stride": stride,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] Wrote {num_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
