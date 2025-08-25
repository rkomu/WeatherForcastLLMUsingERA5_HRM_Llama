# vision_text/train_tinyllama_vision2text.py
# TinyLlama Vision->Text training with captions from either:
#   (A) path,caption
#   (B) date,event description   (auto-detected or override via args)
#
# Adds:
#   - tqdm progress bars
#   - MLflow logging
#   - Periodic evaluation on a validation split
#   - Optional sample generation from soft prompts (logs to MLflow)
#   - QLoRA support (4-bit loading + LoRA on the LM)

import os
import csv
import glob
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# NEW: QLoRA imports
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# local modules
from sat_swin_mae.model import SatSwinMAE
from sat_swin_mae.dataset_era5 import ERA5CubeDataset
from vision_text.vision_adapter import VisionPrefixer

# ============ Optional MLflow ============
_MLFLOW_AVAILABLE = True
try:
    import mlflow
    import mlflow.pytorch
except Exception:
    _MLFLOW_AVAILABLE = False


# ----------------- helpers -----------------
def expand_files(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        hits = glob.glob(p)
        out.extend(hits if hits else [p])
    return sorted(list(dict.fromkeys(out)))


def auto_split_files(files: List[str], val_ratio: float, mode: str, seed: int) -> (List[str], List[str]):
    """Split list of files into train/val by directory groups for stability."""
    assert 0.0 < val_ratio < 1.0
    from collections import defaultdict
    import random

    groups = defaultdict(list)
    for f in files:
        groups[os.path.dirname(f)].append(f)

    train, val = [], []
    rnd = random.Random(seed)
    for _, group in groups.items():
        group = list(group)
        group = sorted(group) if mode == "chronological" else (rnd.shuffle(group) or group)
        n = len(group)
        n_val = max(1, int(round(n * val_ratio)))
        train.extend(group[:-n_val] if n_val < n else group[:1])
        val.extend(group[-n_val:] if n_val < n else group[1:])
    if mode != "chronological":
        rnd.shuffle(train)
        rnd.shuffle(val)
    return train, val


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------- caption loading -----------------
@dataclass
class CaptionTable:
    by_path: Dict[str, str]
    by_date: Dict[pd.Timestamp, str]  # normalized to midnight (date only)
    layout: str  # "path" or "date" or "none"


def _normalize_date_key(x) -> pd.Timestamp:
    ts = pd.to_datetime(x)
    return pd.Timestamp(ts.date())


def load_captions(csv_path: Optional[str],
                  date_col_override: Optional[str] = None,
                  text_col_override: Optional[str] = None) -> CaptionTable:
    if not csv_path:
        return CaptionTable({}, {}, "none")

    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}

    # Explicit overrides
    if date_col_override and text_col_override:
        dcol = date_col_override
        tcol = text_col_override
        if dcol not in df.columns or tcol not in df.columns:
            raise ValueError(
                f"{csv_path} missing overridden columns: {dcol!r}, {tcol!r}. Found: {list(df.columns)}"
            )
        by_date = {}
        for _, row in df.iterrows():
            key = _normalize_date_key(row[dcol])
            by_date[key] = str(row[tcol]).strip()
        return CaptionTable({}, by_date, "date")

    # Auto: path/caption
    if "path" in cols_lower and "caption" in cols_lower:
        path_col = cols_lower["path"]
        cap_col = cols_lower["caption"]
        by_path = {}
        for _, row in df.iterrows():
            p = os.path.normpath(str(row[path_col]))
            c = str(row[cap_col]).strip()
            by_path[p] = c
        return CaptionTable(by_path, {}, "path")

    # Auto: date + text-like col
    candidate_date_names = ["date"]
    candidate_text_names = ["event description", "description", "event", "text", "caption"]
    date_col = next((cols_lower[k] for k in candidate_date_names if k in cols_lower), None)
    text_col = next((cols_lower[k] for k in candidate_text_names if k in cols_lower), None)

    if date_col and text_col:
        by_date = {}
        for _, row in df.iterrows():
            key = _normalize_date_key(row[date_col])
            by_date[key] = str(row[text_col]).strip()
        return CaptionTable({}, by_date, "date")

    # Better error
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fcols = reader.fieldnames or []
    raise ValueError(
        f"{csv_path} must have columns either ('path','caption') or ('date','event description'). "
        f"Found: {list(df.columns)}"
    )


# ----------------- dataset -----------------
@dataclass
class V2TArgs:
    files: List[str]
    variables: List[str]
    window: Dict[str, int]
    stride: Dict[str, int]
    time_start: Optional[str]
    time_end: Optional[str]
    captions: CaptionTable
    drop_if_no_caption: bool
    tokenizer: AutoTokenizer
    max_target_len: int
    anchor: str  # "first" | "middle" | "last"


class ERA5CaptionWindows(Dataset):
    """
    Wraps ERA5CubeDataset and assigns a caption per window:
      - If captions.layout == "path": use a global caption matched by file path/basename.
      - If captions.layout == "date": match window's anchor timestamp to CSV 'date'.
    """
    def __init__(self, a: V2TArgs):
        self.inner = ERA5CubeDataset(
            a.files, a.variables, a.window, a.stride,
            time_start=a.time_start, time_end=a.time_end,
        )
        self.tok = a.tokenizer
        self.max_len = a.max_target_len
        self.drop = a.drop_if_no_caption
        self.anchor = a.anchor
        self.captions = a.captions

        # Prepare samples of (window_index, token_ids)
        self.samples: List[Tuple[int, torch.Tensor]] = []

        if self.captions.layout == "path":
            global_cap = None
            try:
                enc = getattr(self.inner.ds, "encoding", {})
                src = enc.get("source", None)
                if isinstance(src, str):
                    src = [src]
                if src:
                    for fp in src:
                        n = os.path.normpath(fp)
                        if n in self.captions.by_path:
                            global_cap = self.captions.by_path[n]
                            break
                        b = os.path.basename(n)
                        for k, v in self.captions.by_path.items():
                            if os.path.basename(k) == b:
                                global_cap = v
                                break
                        if global_cap:
                            break
            except Exception:
                pass
            if global_cap is None and len(self.captions.by_path):
                global_cap = next(iter(self.captions.by_path.values()))

            if global_cap is None and self.drop:
                self.samples = []
                return

            token_ids = torch.tensor(
                self.tok.encode(global_cap or "", truncation=True, max_length=self.max_len, add_special_tokens=True),
                dtype=torch.long
            )
            self.samples = [(i, token_ids) for i in range(len(self.inner))]
            print(f"[captions] layout=path  applied 1 caption to {len(self.samples)} windows.")
            return

        # layout == "date"
        times = pd.to_datetime(self.inner.data["time"].values)  # type: ignore[attr-defined]
        T = int(self.inner.data.sizes["time"])                  # type: ignore[attr-defined]
        win_T = a.window["T"]

        def anchor_index(t0: int) -> int:
            if self.anchor == "first":
                return t0
            if self.anchor == "middle":
                return min(T - 1, t0 + max(0, win_T // 2))
            if self.anchor == "last":
                return min(T - 1, t0 + max(0, win_T - 1))
            return t0

        matched = 0
        for i in range(len(self.inner)):
            t0, _y, _x = self.inner.idxs[i]  # (t,y,x)
            ta = anchor_index(t0)
            date_key = _normalize_date_key(times[ta])
            cap = self.captions.by_date.get(date_key, None)
            if cap is None and self.drop:
                continue
            token_ids = torch.tensor(
                self.tok.encode(cap or "", truncation=True, max_length=self.max_len, add_special_tokens=True),
                dtype=torch.long
            )
            self.samples.append((i, token_ids))
            if cap is not None:
                matched += 1

        print(f"[captions] layout=date  matched={matched} / {len(self.samples)} usable windows "
              f"(drop_if_no_caption={self.drop})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        wi, token_ids = self.samples[i]
        cube, valid = self.inner[wi]  # cube: (C,T,H,W), valid: (T,H,W)
        return cube, valid, token_ids


def collate_v2t(batch, pad_id: int):
    cubes, valids, ids = zip(*batch)
    cubes = torch.stack(cubes, dim=0)   # (B,C,T,H,W)
    valids = torch.stack(valids, dim=0) # (B,T,H,W)
    maxL = max(x.numel() for x in ids)
    out = torch.full((len(ids), maxL), pad_id, dtype=torch.long)
    for i, seq in enumerate(ids):
        out[i, :seq.numel()] = seq
    return cubes, valids, out


# ----------------- model wrapper -----------------
class TinyLlamaV2T(nn.Module):
    def __init__(self, mae, prefixer, lm, n_latents, pad_id, freeze_lm: bool = False):
        super().__init__()
        self.mae = mae.eval()
        for p in self.mae.parameters():
            p.requires_grad = False
        self.prefixer = prefixer
        self.lm = lm
        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
        self.n_latents = n_latents
        self.pad_id = pad_id
        # LM compute dtype: try embedding weight dtype (works with 4-bit/LoRA), fallback to bf16/fp16/cpu
        try:
            self.lm_dtype = self.lm.get_input_embeddings().weight.dtype
        except Exception:
            self.lm_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )

    def forward(self, cubes: torch.Tensor, _valids: torch.Tensor, input_ids: torch.Tensor):
        # Prefixer calls MAE internally; cubes: (B,C,T,H,W)
        assert cubes.ndim == 5, f"expected (B,C,T,H,W), got {tuple(cubes.shape)}"
        soft = self.prefixer(cubes).to(self.lm_dtype)  # (B, n_latents, d_model)

        tok_emb = self.lm.get_input_embeddings()
        text_emb = tok_emb(input_ids).to(self.lm_dtype)

        inputs_embeds = torch.cat([soft, text_emb], dim=1)
        attn = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=inputs_embeds.device)

        B, L_txt = input_ids.shape
        total_L = self.n_latents + L_txt
        labels = torch.full((B, total_L), -100, dtype=torch.long, device=inputs_embeds.device)
        labels[:, self.n_latents:self.n_latents+L_txt] = input_ids
        labels[:, self.n_latents] = -100

        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels).loss

    @torch.no_grad()
    def generate_from_cubes(self, cubes: torch.Tensor, max_new_tokens: int, eos_token_id: int,
                            temperature: float = 0.9, top_p: float = 0.9):
        self.eval()
        assert cubes.ndim == 5, f"expected (B,C,T,H,W), got {tuple(cubes.shape)}"
        soft = self.prefixer(cubes).to(self.lm_dtype)
        attn = torch.ones(soft.size()[:2], dtype=torch.long, device=soft.device)
        return self.lm.generate(
            inputs_embeds=soft,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id
        )

# ----------------- args -----------------
def build_argparser():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--window_T", type=int, default=8)
    ap.add_argument("--window_H", type=int, default=64)
    ap.add_argument("--window_W", type=int, default=64)
    ap.add_argument("--stride_T", type=int, default=8)
    ap.add_argument("--stride_H", type=int, default=64)
    ap.add_argument("--stride_W", type=int, default=64)
    ap.add_argument("--time_start", type=str, default=None)
    ap.add_argument("--time_end", type=str, default=None)

    ap.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of files for validation if using --files")
    ap.add_argument("--split_mode", type=str, default="chronological", choices=["chronological", "random"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--caption_csv", type=str, default=None)
    ap.add_argument("--drop_if_no_caption", action="store_true")
    ap.add_argument("--caption_date_col", type=str, default=None)
    ap.add_argument("--caption_text_col", type=str, default=None)
    ap.add_argument("--anchor", type=str, choices=["first", "middle", "last"], default="first")

    # training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default=default_device())
    ap.add_argument("--out_dir", type=str, default="checkpoints_v2t")

    # tokenizer / LM
    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--max_target_len", type=int, default=128)
    ap.add_argument("--train_lm", action="store_true", help="Full-finetune LM (ignored if --use_qlora)")

    # adapter
    ap.add_argument("--n_latents", type=int, default=32)
    ap.add_argument("--adapter_layers", type=int, default=2)
    ap.add_argument("--adapter_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)

    # MAE
    ap.add_argument("--mae_ckpt", type=str, required=True)

    # evaluation / logging
    ap.add_argument("--eval_every", type=int, default=1, help="Evaluate validation loss every N epochs (0=only final)")
    ap.add_argument("--gen_samples", type=int, default=3, help="How many samples to generate (and log) at each eval")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64)
    ap.add_argument("--gen_temperature", type=float, default=0.9)
    ap.add_argument("--gen_top_p", type=float, default=0.9)

    # MLflow
    ap.add_argument("--disable_mlflow", action="store_true")
    ap.add_argument("--mlflow_tracking_uri", type=str, default=None)
    ap.add_argument("--mlflow_experiment_name", type=str, default="tinyllama_vision2text")
    ap.add_argument("--mlflow_run_name", type=str, default=None)
    ap.add_argument("--mlflow_tags", nargs="+", default=None)
    ap.add_argument("--log_model_every_n_epochs", type=int, default=0, help="0=only final")

    # ---------------- QLoRA options ----------------
    ap.add_argument("--use_qlora", action="store_true", help="Enable 4-bit loading and LoRA fine-tuning")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", nargs="+", default=[
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ])
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4","fp4"])
    ap.add_argument("--bnb_double_quant", action="store_true", help="Use double quantization in 4-bit")
    ap.add_argument("--grad_checkpointing", action="store_true", help="Enable gradient checkpointing for LM")

    return ap


# ----------------- main -----------------
def make_loader(files, variables, window, stride, batch_size, shuffle, time_start=None, time_end=None,
                tokenizer=None, max_target_len=None, captions: CaptionTable | None = None, drop_if_no_caption=False,
                anchor="first"):
    ds = ERA5CaptionWindows(V2TArgs(
        files=files, variables=variables, window=window, stride=stride,
        time_start=time_start, time_end=time_end,
        captions=captions or CaptionTable({}, {}, "none"),
        drop_if_no_caption=drop_if_no_caption,
        tokenizer=tokenizer, max_target_len=max_target_len or 128,
        anchor=anchor
    ))
    if len(ds) <= 0:
        raise SystemExit("Dataset produced 0 samples. Check time range, captions matching, window/stride.")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2,
                      collate_fn=lambda b: collate_v2t(b, pad_id=tokenizer.pad_token_id))


def evaluate(model: TinyLlamaV2T, loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for cubes, valids, input_ids in loader:
            cubes = cubes.to(device, non_blocking=True)
            valids = valids.to(device, non_blocking=True)
            input_ids = input_ids.to(device)
            loss = model(cubes, valids, input_ids)
            total += float(loss.item()) * cubes.size(0)
            count += cubes.size(0)
    return total / max(1, count)


def sample_and_decode(model: TinyLlamaV2T, loader: DataLoader, tok: AutoTokenizer, args, device: str, n_samples: int) -> List[str]:
    texts = []
    it = iter(loader)
    with torch.no_grad():
        for _ in range(n_samples):
            try:
                cubes, valids, _ = next(it)
            except StopIteration:
                break
            cubes = cubes[:1].to(device)  # one sample
            gen_ids = model.generate_from_cubes(
                cubes,
                max_new_tokens=args.gen_max_new_tokens,
                eos_token_id=tok.eos_token_id or tok.pad_token_id,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p
            )
            text = tok.decode(gen_ids[0], skip_special_tokens=True)
            texts.append(text)
    return texts

def _infer_swin3d_windows_from_ckpt(state_dict) -> tuple[int, int, int] | None:
    """
    Inspect a SatSwinMAE checkpoint state_dict and infer (Wt,Wh,Ww) by
    using:
      N = Wt*Wh*Ww  from '...attn.relative_position_index' which is [N,N]
      S = (2Wt-1)*(2Wh-1)*(2Ww-1) from '...attn.relative_position_bias_table' which is [S, heads]
    Returns (Wt,Wh,Ww) or None if not found.
    """
    N = None
    S = None
    for k, v in state_dict.items():
        if k.endswith("attn.relative_position_index") and hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == v.shape[1]:
            N = int(v.shape[0])
            break
    for k, v in state_dict.items():
        if k.endswith("attn.relative_position_bias_table") and hasattr(v, "shape") and len(v.shape) >= 1:
            S = int(v.shape[0])
            break
    if N is None or S is None:
        return None

    # brute force small reasonable 3D window sizes
    for Wt in range(1, 17):
        for Wh in range(1, 33):
            for Ww in range(1, 33):
                if Wt * Wh * Ww != N:
                    continue
                if (2 * Wt - 1) * (2 * Wh - 1) * (2 * Ww - 1) != S:
                    continue
                return (Wt, Wh, Ww)
    return None


def _strip_relpos_keys(sd: dict) -> dict:
    """Return a copy of sd without relative-position keys (index/bias)."""
    sd = dict(sd)  # shallow copy
    todel = [k for k in sd if "relative_position_bias_table" in k or "relative_position_index" in k]
    for k in todel:
        sd.pop(k, None)
    return sd

def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Files & split
    all_files = expand_files(args.files)
    if len(all_files) < 2:
        raise SystemExit(f"Need at least 2 files to split train/val; got {len(all_files)}")

    train_files, val_files = auto_split_files(all_files, args.val_ratio, args.split_mode, args.seed)
    print(f"[split] train_files={len(train_files)}  val_files={len(val_files)}")

    window = {"T": args.window_T, "H": args.window_H, "W": args.window_W}
    stride = {"T": args.stride_T, "H": args.stride_H, "W": args.stride_W}

    # Tokenizer / LM
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token

    # ---------------- Load LM (standard or QLoRA) ----------------
    if args.use_qlora:
        bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        )
        device_map = None
        # place whole model on a single device if user specified one CUDA device
        if isinstance(args.device, str) and args.device.startswith("cuda"):
            device_map = {"": args.device}

        lm = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # k-bit prep + optional grad checkpointing
        lm = prepare_model_for_kbit_training(lm)
        if args.grad_checkpointing:
            lm.gradient_checkpointing_enable()
            try:
                lm.config.use_cache = False
            except Exception:
                pass

        # LoRA
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )
        lm = get_peft_model(lm, lora_cfg)
        lm.print_trainable_parameters()
        # Important: train mode for LoRA
        lm.train()
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(args.device)
        lm.eval()
        if args.train_lm:
            lm.train()
            for p in lm.parameters():
                p.requires_grad = True

    # Captions
    captions = load_captions(
        args.caption_csv,
        date_col_override=args.caption_date_col,
        text_col_override=args.caption_text_col
    )

    # Build loaders
    train_loader = make_loader(
        train_files, args.variables, window, stride, args.batch_size, True,
        time_start=args.time_start, time_end=args.time_end,
        tokenizer=tok, max_target_len=args.max_target_len,
        captions=captions, drop_if_no_caption=args.drop_if_no_caption,
        anchor=args.anchor
    )
    val_loader = make_loader(
        val_files, args.variables, window, stride, args.batch_size, False,
        time_start=args.time_start, time_end=args.time_end,
        tokenizer=tok, max_target_len=args.max_target_len,
        captions=captions, drop_if_no_caption=args.drop_if_no_caption,
        anchor=args.anchor
    )

    # MAE & adapter
    print(train_loader)
    in_chans = int(train_loader.dataset.inner.C)  # type: ignore[attr-defined]

    c, v, y = next(iter(train_loader))
    print("sample shapes:", c.shape, v.shape, y.shape)  # expect (B,C,T,H,W), (B,T,H,W), (B,L)

    # --- Load MAE checkpoint, infer correct Swin 3D window sizes, then build model accordingly ---
    # Safer torch.load regarding future weights_only default flip:
    try:
        ckpt = torch.load(args.mae_ckpt, map_location="cpu", weights_only=True)  # PyTorch >=2.4
    except TypeError:
        ckpt = torch.load(args.mae_ckpt, map_location="cpu")  # fallback for older PyTorch

    # Infer window size from the checkpoint
    ws = _infer_swin3d_windows_from_ckpt(ckpt)
    if ws is None:
        # Sensible default that matches many SatSwinMAE trainings (and matches your error message shapes)
        ws = (2, 8, 8)
    print(f"[SatSwinMAE] Using window_size={ws} (inferred from checkpoint)")

    mae = SatSwinMAE(
        in_chans=in_chans, out_chans=in_chans,
        embed_dim=96, depths=(2, 2), num_heads=(3, 6),
        window_size=ws, patch_size=(2, 4, 4), mask_ratio=0.75
    )

    # First try a straight load
    try:
        missing, unexpected = mae.load_state_dict(ckpt, strict=False)
    except RuntimeError as e:
        print(f"[SatSwinMAE] load_state_dict encountered mismatch: {e}\n"
            f"[SatSwinMAE] Retrying after stripping relative-position keys ...")
        ckpt2 = _strip_relpos_keys(ckpt)
        missing, unexpected = mae.load_state_dict(ckpt2, strict=False)

    mae = mae.to(args.device).eval()
    for p in mae.parameters():
        p.requires_grad = False

    # Derive d_vis robustly by a tiny forward with zeros
    with torch.no_grad():
        dummy = torch.zeros(1, in_chans, args.window_T, args.window_H, args.window_W, device=args.device)
        vis_tokens, _ = mae.encode_tokens(dummy)
        d_vis = int(vis_tokens.shape[-1])

    d_model = int(lm.config.hidden_size)
    prefixer = VisionPrefixer(
        mae_encoder=mae,
        d_vis=d_vis, d_model=d_model,
        n_latents=args.n_latents, n_layers=args.adapter_layers,
        n_heads=args.adapter_heads, dropout=args.dropout
    ).to(args.device)

    # Freeze LM only if not training (no LoRA and no full-FT)
    freeze_lm_flag = not (args.use_qlora or args.train_lm)
    model = TinyLlamaV2T(mae, prefixer, lm, n_latents=args.n_latents, pad_id=tok.pad_token_id,
                         freeze_lm=freeze_lm_flag).to(args.device if not args.use_qlora else args.device)

    # Optimizer (adapter always; plus LM params that require_grad—LoRA if QLoRA)
    params = list(prefixer.parameters()) + [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    # ------------- MLflow init -------------
    use_mlflow = (not args.disable_mlflow) and _MLFLOW_AVAILABLE
    if use_mlflow and args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    if use_mlflow:
        mlflow.set_experiment(args.mlflow_experiment_name)
        run = mlflow.start_run(run_name=args.mlflow_run_name)
        # tags
        mlflow.set_tag("model_type", "TinyLlama_Vision2Text")
        mlflow.set_tag("framework", "PyTorch")
        mlflow.set_tag("task", "vision-to-text")
        mlflow.set_tag("use_qlora", str(args.use_qlora))
        if args.mlflow_tags:
            for tag in args.mlflow_tags:
                if "=" in tag:
                    k, v = tag.split("=", 1)
                    mlflow.set_tag(k.strip(), v.strip())
        # params
        mlflow.log_params({
            "variables": args.variables,
            "window_T": args.window_T, "window_H": args.window_H, "window_W": args.window_W,
            "stride_T": args.stride_T, "stride_H": args.stride_H, "stride_W": args.stride_W,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "n_latents": args.n_latents, "adapter_layers": args.adapter_layers,
            "adapter_heads": args.adapter_heads, "dropout": args.dropout,
            "model_name": args.model_name, "max_target_len": args.max_target_len,
            "train_lm": args.train_lm,
            "val_ratio": args.val_ratio, "split_mode": args.split_mode, "seed": args.seed,
            "time_start": args.time_start, "time_end": args.time_end,
            "gen_samples": args.gen_samples, "gen_max_new_tokens": args.gen_max_new_tokens,
            "gen_temperature": args.gen_temperature, "gen_top_p": args.gen_top_p,
            "eval_every": args.eval_every,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "lora_target_modules": ",".join(args.lora_target_modules),
            "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
            "bnb_double_quant": args.bnb_double_quant,
            "grad_checkpointing": args.grad_checkpointing
        })
        # dataset info
        mlflow.log_params({
            "train_files_count": len(train_files),
            "val_files_count": len(val_files),
            "in_chans": in_chans,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "d_vis": d_vis,
            "d_model": d_model
        })

    # ------------- Training loop -------------
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for cubes, valids, input_ids in pbar:
            cubes = cubes.to(args.device, non_blocking=True)
            valids = valids.to(args.device, non_blocking=True)
            input_ids = input_ids.to(args.device)

            loss = model(cubes, valids, input_ids)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            running += float(loss.item())
            count_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = running / max(1, count_batches)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_train:.4f}")

        # Log train loss
        if use_mlflow:
            mlflow.log_metric("train_loss", avg_train, step=epoch)

        # Save adapter checkpoint (VisionPrefixer)
        ckpt_path = os.path.join(args.out_dir, f"adapter_epoch{epoch}.pt")
        torch.save(prefixer.state_dict(), ckpt_path)
        if use_mlflow and (args.log_model_every_n_epochs > 0 and epoch % args.log_model_every_n_epochs == 0):
            try:
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            except Exception as e:
                print(f"[MLflow] Warning: failed to log checkpoint: {e}")

        # If QLoRA, also save LoRA adapter each epoch (small)
        if args.use_qlora:
            lora_dir = os.path.join(args.out_dir, f"lora_epoch{epoch}")
            os.makedirs(lora_dir, exist_ok=True)
            try:
                model.lm.save_pretrained(lora_dir)
                tok.save_pretrained(lora_dir)
            except Exception as e:
                print(f"[LoRA] Warning: failed to save LoRA at epoch {epoch}: {e}")

        # ---------- EVAL ----------
        do_eval = (args.eval_every > 0 and (epoch % args.eval_every == 0)) or (epoch == args.epochs)
        if do_eval:
            val_loss = evaluate(model, val_loader, args.device)
            print(f"[eval] epoch={epoch}  val_loss={val_loss:.4f}")

            # Track best
            best_val = min(best_val, val_loss)

            if use_mlflow:
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("best_val_loss", best_val, step=epoch)

                # Sample generations
                if args.gen_samples > 0:
                    try:
                        gens = sample_and_decode(model, val_loader, tok, args, args.device, args.gen_samples)
                        # Log as a single text artifact
                        text_blob = "\n\n".join([f"### Sample {i+1}\n{t}" for i, t in enumerate(gens)])
                        mlflow.log_text(text_blob, artifact_file=f"samples/epoch_{epoch}.md")
                    except Exception as e:
                        print(f"[MLflow] Warning: failed to log generations: {e}")

        # Optional: log final model (adapter only)
        if use_mlflow and (epoch == args.epochs) and (args.log_model_every_n_epochs == 0):
            try:
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            except Exception as e:
                print(f"[MLflow] Warning: failed to log final checkpoint: {e}")

    # Save final adapters
    final_path = os.path.join(args.out_dir, "adapter_final.pt")
    torch.save(prefixer.state_dict(), final_path)

    if args.use_qlora:
        lora_final_dir = os.path.join(args.out_dir, "lora_final")
        os.makedirs(lora_final_dir, exist_ok=True)
        try:
            model.lm.save_pretrained(lora_final_dir)
            tok.save_pretrained(lora_final_dir)
        except Exception as e:
            print(f"[LoRA] Warning: failed to save final LoRA: {e}")

    if use_mlflow:
        try:
            mlflow.log_artifact(final_path, artifact_path="checkpoints")
        except Exception as e:
            print(f"[MLflow] Warning: failed to log final adapter: {e}")
        mlflow.end_run()


if __name__ == "__main__":
    main()
