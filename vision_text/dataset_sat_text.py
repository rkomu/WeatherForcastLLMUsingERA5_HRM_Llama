
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List
from sat_swin_mae.dataset_era5 import ERA5CubeDataset
import glob
import pandas as pd
from typing import Optional, List, Dict, Tuple

def tokenize_plain(tok, text: str):
    # Prefer HF-style call API (works for fast/slow tokenizers)
    if hasattr(tok, "__call__"):
        out = tok(text, add_special_tokens=False, return_attention_mask=False)
        # transformers returns dict; extract list[int]
        if isinstance(out, dict) and "input_ids" in out:
            return out["input_ids"]
    # Fallback to .encode
    if hasattr(tok, "encode"):
        try:
            return tok.encode(text, add_special_tokens=False)
        except TypeError:
            return tok.encode(text)  # for tokenizers without that arg
    raise TypeError("Tokenizer must be callable or have .encode()")

def _auto_caption(cube: np.ndarray, chan_names: Optional[List[str]]=None) -> str:
    C,T,H,W = cube.shape
    def find(name):
        if not chan_names: return None
        for i, n in enumerate(chan_names):
            if name in n: return i
        return None
    parts = []
    for key in ["t2m", "sp", "u10", "v10", "tp", "ssrd"]:
        idx = find(key)
        if idx is not None:
            x = float(np.nanmean(cube[idx]))
            parts.append(f"{key}~{x:.2f}")
    summary = ", ".join(parts) if parts else f"C={C}, T={T}"
    return f"Satellite ERA5 window: {summary}."

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

class ERA5CaptionDatasetCSV(Dataset):
    """
    Use a CSV that has at least two columns: `date` and `event description` (or `event_description`).
    For each ERA5 window, we take the **anchor time** = the last timestep in the window,
    convert to UTC date (YYYY-MM-DD), and pair that window with *all* captions from that date.
    This expands the dataset length so each (window, caption) is a training item.

    Args:
        csv_path: path to the CSV
        drop_if_no_caption: if True, skip windows with no matching captions; else fall back to auto caption
        anchor: "last" (default), "first", or "middle" timestep of the window to compute the date
    """
    def __init__(
        self,
        files, variables, window, stride,
        tokenizer,
        csv_path: str,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        max_len: int = 128,
        drop_if_no_caption: bool = True,
        anchor: str = "last",
    ):
        all_files = expand_files(files)
        self.inner = ERA5CubeDataset(all_files, variables, window, stride,
                                     time_start=time_start, time_end=time_end)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.drop_if_no_caption = drop_if_no_caption
        self.anchor = anchor

        # ---- read CSV and build date -> [captions] map ----
        df = pd.read_csv(csv_path)
        # normalize column names
        cols = {c.lower().strip(): c for c in df.columns}
        if "date" not in cols:
            raise ValueError("CSV must have a 'date' column.")
        # accept either 'event description' or 'event_description'
        desc_col = cols.get("event description", cols.get("event_description", None))
        if desc_col is None:
            # if user named it differently, show available
            raise ValueError(f"CSV needs an 'event description' column (or 'event_description'). Found: {list(df.columns)}")

        # parse date to pandas datetime (UTC-naive dates OK)
        df["__date__"] = pd.to_datetime(df[cols["date"]]).dt.date
        df["__text__"] = df[desc_col].astype(str)
        groups: Dict[pd.Timestamp, List[str]] = {}
        for d, sub in df.groupby("__date__"):
            groups[d] = sub["__text__"].tolist()
        self.captions_by_date = groups

        # ---- precompute index mapping: (window_idx, caption_idx) ----
        # grab the xarray time coordinate
        time_name = "valid_time"
        if hasattr(self.inner.data, "sizes") and "valid_time" not in self.inner.data.sizes:
            # fallback to any known name used earlier
            for cand in ["time", "valid_time", "forecast_time", "analysis_time", "initial_time"]:
                if cand in self.inner.data.sizes:
                    time_name = cand
                    break
        times = pd.to_datetime(self.inner.data[time_name].values)  # ndarray of datetimes

        self._items: List[Tuple[int, int]] = []  # (window_idx, caption_idx in that date list)
        self._date_for_window: List[pd.Timestamp] = []

        T_w = window["T"]
        for wi, (t0, y, x) in enumerate(self.inner.idxs):
            if self.anchor == "first":
                ti = t0
            elif self.anchor == "middle":
                ti = t0 + (T_w // 2)
            else:  # "last"
                ti = t0 + T_w - 1
            if ti >= len(times):
                # safe guard; skip invalid window (shouldn't happen)
                continue
            d = times[ti].date()  # datetime.date
            caps = self.captions_by_date.get(d, None)
            if not caps:
                if self.drop_if_no_caption:
                    continue
                else:
                    # one synthetic caption
                    self._items.append((wi, -1))
                    self._date_for_window.append(d)
            else:
                for ci in range(len(caps)):
                    self._items.append((wi, ci))
                    self._date_for_window.append(d)

        if len(self._items) == 0:
            raise ValueError("After pairing with CSV captions, no training items were produced. "
                             "Consider setting drop_if_no_caption=False or adjusting time range.")

        # keep channel names for optional auto-captions
        self._chan_names = getattr(self.inner, "chan_names", None)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        wi, ci = self._items[idx]
        cube, valid = self.inner[wi]  # (C,T,H,W)

        # choose caption text
        if ci >= 0:
            d = self._date_for_window[idx]
            texts = self.captions_by_date[d]
            text = texts[ci]
        else:
            # fallback synthetic caption
            text = _auto_caption(cube.numpy(), self._chan_names)

        # tokenize
        tok = self.tokenizer
        ids = tokenize_plain(self.tokenizer, text)
        bos = getattr(tok, 'bos_id', None) or getattr(tok, 'bos_token_id', None)
        eos = getattr(tok, 'eos_id', None) or getattr(tok, 'eos_token_id', None)
        if bos is not None:
            ids = [bos] + ids
        if eos is not None:
            ids = ids + [eos]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            if eos is not None:
                ids[-1] = eos

        return cube, valid, torch.tensor(ids, dtype=torch.long)