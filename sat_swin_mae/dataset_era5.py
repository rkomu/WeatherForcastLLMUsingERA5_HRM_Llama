import torch
from torch.utils.data import Dataset
import numpy as np

try:
    import xarray as xr
except Exception:
    xr = None

class ERA5CubeDataset(Dataset):
    """
    Builds cubes (C, T, H, W) from ERA5 NetCDF/GRIB files using xarray.
    Provide:
      - files: list of paths
      - variables: list of variable names (e.g., ["u10","v10","tp","t2m","sp"])
      - window: {"T":24, "H":64, "W":64}
      - stride: {"T":24, "H":32, "W":32}
      - norm_stats: optional {var: (mean, std)}
    """
    def __init__(self, files, variables, window, stride, norm_stats=None):
        assert xr is not None, "xarray is required to load ERA5 files"
        self.ds = xr.open_mfdataset(files, combine='by_coords')
        print(f"Loaded dataset with variables: {list(self.ds.variables)}")
        for v in variables:
            if v not in self.ds:
                raise ValueError(f"Variable {v} not found in dataset")
        self.variables = variables
        self.window = window
        self.stride = stride
        self.norm_stats = norm_stats or {}

        da = xr.concat([self.ds[v] for v in variables], dim="var")
        print(da)
        da = da.transpose("valid_time", "var", "latitude", "longitude")  # (T, C, H, W)
        self.data = da

        self.T = self.data.sizes["valid_time"]
        self.C = self.data.sizes["var"]
        self.H = self.data.sizes["latitude"]
        self.W = self.data.sizes["longitude"]

        print(f"ERA5CubeDataset: T={self.T}, H={self.H}, W={self.W}, window_T={window['T']}, window_H={window['H']}, window_W={window['W']}")
        if self.T < window['T'] or self.H < window['H'] or self.W < window['W']:
            print("Warning: Window size is larger than data dimension. No samples will be generated.")
        self.idxs = []
        for t in range(0, self.T - window["T"] + 1, stride["T"]):
            for y in range(0, self.H - window["H"] + 1, stride["H"]):
                for x in range(0, self.W - window["W"] + 1, stride["W"]):
                    self.idxs.append((t,y,x))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        t, y, x = self.idxs[idx]
        sl = dict(
            valid_time=slice(t, t + self.window["T"]),
            var=slice(0, self.C),
            latitude=slice(y, y + self.window["H"]),
            longitude=slice(x, x + self.window["W"]),
        )
        cube = self.data.isel(**sl).values.astype(np.float32)  # (T, C, H, W)
        cube = np.transpose(cube, (1, 0, 2, 3))               # (C, T, H, W)

        # validity mask: True where ALL variables are finite at that (t,h,w)
        valid = np.isfinite(cube).all(axis=0, keepdims=True)  # (1, T, H, W)

        # per-var normalize, ignoring NaNs/Infs; fill NaNs with per-var mean BEFORE z-score
        for i, v in enumerate(self.variables):
            arr = cube[i]
            m = np.nanmean(arr)
            s = np.nanstd(arr) + 1e-6
            arr = np.nan_to_num(arr, nan=m, posinf=m, neginf=m)
            cube[i] = (arr - m) / s

        # optional: clip extreme z-scores to avoid occasional outliers
        np.clip(cube, -8.0, 8.0, out=cube)

        return torch.from_numpy(cube), torch.from_numpy(valid.astype(np.bool_))
