import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedCubeMemmapDataset(Dataset):
    """
    Dataset that reads pre-normalized ERA5 cubes from a memmap cache produced by
    tools/cache_era5_cubes.py. Avoids re-opening NetCDF files during training.
    """

    def __init__(self, cache_dir: str | os.PathLike):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Expected metadata at {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta
        self.num_samples = int(meta["num_samples"])
        self.cube_shape = tuple(meta["cube_shape"])  # (C,T,H,W)
        self.valid_shape = tuple(meta["valid_shape"])  # (T,H,W)
        self.dtype = np.dtype(meta.get("cube_dtype", "float32"))
        self.valid_dtype = np.dtype(meta.get("valid_dtype", "bool"))
        self.chan_names = meta.get("chan_names", None)
        self.window = meta.get("window", None)
        self.stride = meta.get("stride", None)
        self.C = self.cube_shape[0]
        self.T = self.cube_shape[1]
        self.H = self.cube_shape[2]
        self.W = self.cube_shape[3]
        self._cube_mem: Optional[np.memmap] = None
        self._valid_mem: Optional[np.memmap] = None

    def __len__(self):
        return self.num_samples

    def _ensure_memmaps(self):
        if self._cube_mem is None:
            self._cube_mem = np.memmap(
                self.cache_dir / "cubes.bin",
                dtype=self.dtype,
                mode="r",
                shape=(self.num_samples,) + self.cube_shape,
            )
        if self._valid_mem is None:
            self._valid_mem = np.memmap(
                self.cache_dir / "valid_mask.bin",
                dtype=self.valid_dtype,
                mode="r",
                shape=(self.num_samples,) + self.valid_shape,
            )

    def __getitem__(self, idx):
        self._ensure_memmaps()
        cube_np = np.array(self._cube_mem[idx], copy=False)
        valid_np = np.array(self._valid_mem[idx], copy=False)
        cube = torch.from_numpy(cube_np)
        valid = torch.from_numpy(valid_np)
        return cube, valid
