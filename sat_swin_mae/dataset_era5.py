# sat_swin_mae/dataset_era5.py
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

try:
    import xarray as xr
except Exception:
    xr = None

TIME_CANDIDATES = ["time", "valid_time", "forecast_time", "analysis_time", "initial_time"]
LAT_CANDIDATES  = ["latitude", "lat", "y"]
LON_CANDIDATES  = ["longitude", "lon", "x"]
LEVEL_CANDIDATES = ["pressure_level", "level"]

def _pick_dim_name(obj, candidates):
    sizes = getattr(obj, "sizes", {})
    for n in candidates:
        if n in sizes:
            return n
    coords = getattr(obj, "coords", {})
    for n in candidates:
        if n in coords:
            return n
    dims = getattr(obj, "dims", ())
    try:
        for n in candidates:
            if n in dims:
                return n
    except Exception:
        pass
    return None

def _probe_nc(fp, engine=None):
    try:
        if engine:
            with xr.open_dataset(fp, engine=engine) as ds:
                _ = tuple(ds.sizes.items())
        else:
            with xr.open_dataset(fp) as ds:
                _ = tuple(ds.sizes.items())
        return True, None
    except Exception as e:
        return False, str(e)

class ERA5CubeDataset(Dataset):
    """
    Builds cubes (C, T, H, W) from ERA5 NetCDFs.

    - Filters missing/corrupt files.
    - Supports single-level and pressure-level variables.
    - Optionally aligns all variables to the dataset's common grid.
    - Optional time filtering via time_start/time_end (inclusive).
    - Returns (cube, valid_mask):
        cube:  float32 tensor (C, T, H, W), z-scored per channel
        valid_mask: bool tensor (T, H, W) = True where all channels are finite
    """

    def __init__(
        self,
        files,
        variables,
        window,
        stride,
        norm_stats=None,
        netcdf_engine=None,
        pressure_handling="stack",   # "stack" or "select"
        pressure_levels=None,        # dict var->list/int when pressure_handling="select"
        time_start: str | None = None,
        time_end:   str | None = None,
        align_to_grid: bool = False,  # NEW: regrid variables to common lat/lon grid
        interp_method: str = "nearest",  # 'nearest' (fast) or 'linear'
    ):
        assert xr is not None, "xarray is required"
        self.window = window
        self.stride = stride
        self.variables = variables
        self.norm_stats = norm_stats or {}
        self.netcdf_engine = netcdf_engine
        self.pressure_handling = pressure_handling
        self.pressure_levels = pressure_levels or {}
        self.align_to_grid = align_to_grid
        self.interp_method = interp_method

        # ---------- filter file paths ----------
        paths = [str(p) for p in files]
        print(f"[ERA5CubeDataset] Found {len(paths)} files")
        existing = [p for p in paths if os.path.exists(p)]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            warnings.warn(f"[ERA5CubeDataset] Skipping {len(missing)} missing files (e.g., {missing[:2]})")

        valid_files, bad = [], []
        for fp in existing:
            ok, err = _probe_nc(fp, netcdf_engine)
            if ok:
                valid_files.append(fp)
            else:
                bad.append((fp, err))
        if bad:
            preview = ", ".join([f for f,_ in bad[:3]])
            warnings.warn(f"[ERA5CubeDataset] Skipping {len(bad)} unreadable files (e.g., {preview})")

        if not valid_files:
            raise ValueError("[ERA5CubeDataset] No valid ERA5 files left after filtering.")

        # ---------- open lazily with outer join on time ----------
        open_kwargs = dict(
            combine="by_coords",
            coords="minimal",
            compat="override",
            join="outer",     # allow different time lengths
            parallel=False,
        )
        if netcdf_engine:
            ds = xr.open_mfdataset(valid_files, engine=netcdf_engine, **open_kwargs)
        else:
            ds = xr.open_mfdataset(valid_files, **open_kwargs)

        # ---------- time filter (inclusive) ----------
        time_name = _pick_dim_name(ds, TIME_CANDIDATES)
        if time_name is None:
            raise ValueError(
                f"Could not find a time axis; dims={list(getattr(ds,'sizes',{}).keys())} coords={list(ds.coords)}"
            )
        if time_start or time_end:
            ts = pd.to_datetime(time_start) if time_start else None
            te = pd.to_datetime(time_end)   if time_end   else None
            if ts is not None and te is not None and te < ts:
                raise ValueError(f"time_end ({te}) must be >= time_start ({ts})")
            if ts is not None and te is not None:
                ds = ds.sel({time_name: slice(ts, te)})
            elif ts is not None:
                ds = ds.sel({time_name: slice(ts, None)})
            else:
                ds = ds.sel({time_name: slice(None, te)})

        # ensure monotonic time
        ds = ds.sortby(time_name)

        # ---------- variable sanity ----------
        missing_vars = [v for v in variables if v not in ds]
        if missing_vars:
            raise ValueError(f"Variables not found: {missing_vars}. Available: {list(ds.data_vars)}")

        lat_name = _pick_dim_name(ds, LAT_CANDIDATES)
        lon_name = _pick_dim_name(ds, LON_CANDIDATES)
        if lat_name is None or lon_name is None:
            raise ValueError(f"Missing lat/lon. dims={list(ds.sizes.keys())} coords={list(ds.coords)}")
        target_lat = ds[lat_name]
        target_lon = ds[lon_name]

        # ---------- build channels (time, latitude, longitude) ----------
        chans = []
        chan_names = []

        for v in variables:
            da = ds[v]

            # normalize dim names
            ren = {}
            tdn = _pick_dim_name(da, TIME_CANDIDATES)
            ltn = _pick_dim_name(da, LAT_CANDIDATES)
            lnn = _pick_dim_name(da, LON_CANDIDATES)
            if tdn and tdn != "time": ren[tdn] = "time"
            if ltn and ltn != "latitude": ren[ltn] = "latitude"
            if lnn and lnn != "longitude": ren[lnn] = "longitude"
            if ren:
                da = da.rename(ren)

            # pressure handling
            lvl_name = None
            for cand in LEVEL_CANDIDATES:
                if cand in da.dims:
                    lvl_name = cand
                    break

            def _prep_spatial(d):
                # ensure (time, latitude, longitude)
                d = d.transpose("time", "latitude", "longitude")
                d = d.reset_coords(drop=True)
                # optional regrid to dataset grid
                if self.align_to_grid:
                    if (int(d.sizes["latitude"]) != int(target_lat.size)) or (int(d.sizes["longitude"]) != int(target_lon.size)):
                        d = d.interp(latitude=target_lat, longitude=target_lon, method=self.interp_method)
                return d

            if lvl_name is not None:
                if self.pressure_handling == "select":
                    levels = self.pressure_levels.get(v, None)
                    if levels is None:
                        levels = [int(da[lvl_name].values[0])]
                    elif isinstance(levels, (int, float)):
                        levels = [int(levels)]
                else:
                    levels = [int(x) for x in list(da[lvl_name].values)]

                for L in levels:
                    dL = _prep_spatial(da.sel({lvl_name: L}))
                    chans.append(dL.expand_dims({"var": [f"{v}_pl{L}"]}))
                    chan_names.append(f"{v}_pl{L}")
            else:
                dS = _prep_spatial(da)
                chans.append(dS.expand_dims({"var": [v]}))
                chan_names.append(v)

        da_all = xr.concat(chans, dim="var", coords="minimal", compat="override", join="override")
        da_all = da_all.transpose("time", "var", "latitude", "longitude")

        self.ds = ds
        self.data = da_all
        self.chan_names = chan_names

        # shapes
        self.T = int(self.data.sizes["time"])
        self.C = int(self.data.sizes["var"])
        self.H = int(self.data.sizes["latitude"])
        self.W = int(self.data.sizes["longitude"])

        # ---------- sliding indices with diagnostics ----------
        t_win, h_win, w_win = self.window["T"], self.window["H"], self.window["W"]
        t_str, h_str, w_str = self.stride["T"], self.stride["H"], self.stride["W"]

        if self.T < t_win or self.H < h_win or self.W < w_win:
            raise ValueError(
                "[ERA5CubeDataset] No windows: "
                f"T/H/W={self.T}/{self.H}/{self.W}, "
                f"need at least window_T/H/W={t_win}/{h_win}/{w_win}. "
                "Adjust --time_start/--time_end or window sizes."
            )

        self.idxs = []
        for t in range(0, self.T - t_win + 1, max(1, t_str)):
            for y in range(0, self.H - h_win + 1, max(1, h_str)):
                for x in range(0, self.W - w_win + 1, max(1, w_str)):
                    self.idxs.append((t, y, x))

        if len(self.idxs) == 0:
            raise ValueError(
                "[ERA5CubeDataset] Computed 0 sliding windows. "
                f"T/H/W={self.T}/{self.H}/{self.W}, window={self.window}, stride={self.stride}. "
                "Try smaller strides or smaller window."
            )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        t, y, x = self.idxs[idx]
        sl = dict(
            time=slice(t, t + self.window["T"]),
            var=slice(0, self.C),
            latitude=slice(y, y + self.window["H"]),
            longitude=slice(x, x + self.window["W"]),
        )
        cube = self.data.isel(**sl).values.astype(np.float32)   # (T,C,H,W)
        cube = np.transpose(cube, (1, 0, 2, 3))                 # (C,T,H,W)

        # validity where ALL channels finite
        valid = np.isfinite(cube).all(axis=0)  # (T,H,W)

        # per-channel z-score with NaN-safe fill + clip
        for i in range(cube.shape[0]):
            arr = cube[i]
            m = np.nanmean(arr)
            s = np.nanstd(arr) + 1e-6
            arr = np.nan_to_num(arr, nan=m, posinf=m, neginf=m)
            arr = (arr - m) / s
            np.clip(arr, -8.0, 8.0, out=arr)
            cube[i] = arr

        return torch.from_numpy(cube), torch.from_numpy(valid.astype(np.bool_))
