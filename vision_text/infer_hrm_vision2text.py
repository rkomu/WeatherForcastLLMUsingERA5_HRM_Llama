import argparse, os, torch
from sat_swin_mae.model import SatSwinMAE
from vision_text.vision_adapter import VisionPrefixer
from transformers import AutoTokenizer
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.hrm.hrm_lm_adapter import HRMAsTextLM

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from glob import glob
from datetime import datetime
from sat_swin_mae.dataset_era5 import ERA5CubeDataset
import pandas as pd

def expand_input_files(paths):
    out = []
    for p in paths:
        if any(ch in p for ch in ["*", "?", "["]):  # glob pattern
            out.extend(glob(p, recursive=True))
        elif os.path.isdir(p):                       # a directory
            out.extend(glob(os.path.join(p, "**", "*.nc"), recursive=True))
        else:                                        # a concrete file path
            out.append(p)
    # keep only existing files, de-dup, sort for stability
    out = sorted({f for f in out if os.path.exists(f)})
    return out

def _find_time_coord(xr_ds):
    # Try common names in ERA5/CFGRIB conversions
    for cand in ["time", "valid_time", "forecast_time", "analysis_time", "initial_time"]:
        if cand in getattr(xr_ds, "sizes", {}):
            return cand
    # fallback to first datetime-like coord
    for k in xr_ds.coords:
        try:
            if "datetime64" in str(xr_ds[k].dtype):
                return k
        except Exception:
            pass
    raise KeyError("Could not find a time coordinate in ERA5 dataset.")

def window_indices_for_date(inner_ds, window, target_date_str, anchor="last"):
    """
    Returns a list of indices (into inner_ds.idxs) whose anchor timestep date equals target_date_str (YYYY-MM-DD).
    """
    import pandas as pd
    # Identify the time coordinate in the dataset
    time_name = _find_time_coord(inner_ds.data)
    # Convert dataset time values into pandas datetime objects
    times = pd.to_datetime(inner_ds.data[time_name].values)  # ndarray of datetimes
    # Convert the target date string into a Python date object
    target_date = pd.to_datetime(target_date_str).date()
    T_w = window["T"]  # number of timesteps in each window
    matches = []
    # Iterate over all indexed windows in the dataset
    for wi, (t0, y, x) in enumerate(inner_ds.idxs):
        # Determine which timestep in the window to use as the "anchor"
        if anchor == "first":
            ti = t0  # first timestep
        elif anchor == "middle":
            ti = t0 + (T_w // 2)  # middle timestep
        else:
            ti = t0 + T_w - 1  # last timestep
        # Skip if the anchor index is outside the range of available times
        if ti >= len(times):
            continue
        # Get the date at the chosen anchor timestep
        d = times[ti].date()
        # If the date matches the target date, record the window index
        if d == target_date:
            matches.append(wi)
    # Return all window indices that matched the target date
    return matches

def get_token_embedding_and_dim(hrm):
    emb = getattr(hrm, 'tok_emb', None) or getattr(hrm, 'embed_tokens', None)
    if emb is None:
        raise AttributeError("HRM model must expose .tok_emb or .embed_tokens (nn.Embedding)")
    d_model = getattr(hrm, 'd_model', None) or emb.embedding_dim
    return emb, d_model

def forward_with_embeds(hrm, inputs_embeds, attention_mask=None):
    if hasattr(hrm, 'forward_with_embeds'):
        return hrm.forward_with_embeds(inputs_embeds, attention_mask=attention_mask)
    try:
        return hrm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    except TypeError:
        raise AttributeError("Add forward_with_embeds(self, inputs_embeds, attention_mask=None) to your HRM.")

def load_hrm_and_tokenizer(device):
    # ---- tokenizer (swap to your tokenizer if you have one) ----
    model_name = "gpt2"            # or your own tokenizer path
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # tok.encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    tok.bos_id = getattr(tok, "bos_token_id", None)
    tok.eos_id = getattr(tok, "eos_token_id", tok.eos_token_id)
    vocab_size = len(tok)

    # ---- HRM config (make seq_len >= (n_latents + max_text_len)) ----
    cfg = {
        "batch_size": 32,                  # not critical for core_forward
        "seq_len": 256,                    # >= prompts(M) + max text tokens
        "puzzle_emb_ndim": 0,              # we don't use puzzle embeddings here
        "num_puzzle_identifiers": 1,
        "vocab_size": vocab_size,

        "H_cycles": 2, "L_cycles": 2,
        "H_layers": 4, "L_layers": 4,

        "hidden_size": 512,
        "expansion": 4.0,
        "num_heads": 8,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,

        "halt_max_steps": 16,
        "halt_exploration_prob": 0.0,

        "forward_dtype": "float16",        # safer default on consumer GPUs
    }

    hrm_full = HierarchicalReasoningModel_ACTV1(cfg).to(device).eval()
    hrm_lm = HRMAsTextLM(hrm_full.inner).to(device).eval()
    return hrm_lm, tok

def get_token_embedding_and_dim(hrm):
    emb = getattr(hrm, 'tok_emb', None) or getattr(hrm, 'embed_tokens', None)
    if emb is None:
        raise AttributeError('HRM must expose .tok_emb or .embed_tokens')
    d_model = getattr(hrm, 'd_model', None) or emb.embedding_dim
    return emb, d_model

def forward_with_embeds(hrm, inputs_embeds, attention_mask=None):
    if hasattr(hrm, 'forward_with_embeds'):
        return hrm.forward_with_embeds(inputs_embeds, attention_mask=attention_mask)
    try:
        return hrm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    except TypeError:
        raise AttributeError('Add forward_with_embeds(self, inputs_embeds, attention_mask=None) to your HRM.')

@torch.no_grad()
def generate_greedy(hrm, tok_emb, prompts, tokenizer, max_new_tokens=64, temperature=1.0):
    device = prompts.device
    bos = getattr(tokenizer, 'bos_id', None) or getattr(tokenizer, 'bos_token_id', None) or 1
    eos = getattr(tokenizer, 'eos_id', None) or getattr(tokenizer, 'eos_token_id', None)

    input_ids = torch.tensor([[bos]], device=device, dtype=torch.long).repeat(prompts.size(0),1)
    txt_emb = tok_emb(input_ids)
    seq = torch.cat([prompts, txt_emb], dim=1)

    for _ in range(max_new_tokens):
        out = forward_with_embeds(hrm, seq)
        logits = out[0] if isinstance(out, (tuple,list)) else out
        next_logits = logits[:, -1, :] / max(1e-4, temperature)
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        if eos is not None and (next_id == eos).all():
            break
        seq = torch.cat([seq, tok_emb(next_id)], dim=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    if hasattr(tokenizer, 'decode'):
        return [tokenizer.decode(ids.tolist()) for ids in input_ids]
    return ["<no-decode>"] * input_ids.size(0)

def _guess_time_name(ds):
    # prefer coords, then dims
    candidates = ["valid_time", "time", "forecast_time", "analysis_time", "initial_time"]
    for k in candidates:
        if k in getattr(ds, "coords", {}):
            return k
    for k in candidates:
        if k in getattr(ds, "dims", {}):
            return k
    # fallback: first datetime64-like coord
    for k in ds.coords:
        if "datetime64" in str(ds[k].dtype):
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mae_ckpt', type=str, required=True)
    ap.add_argument('--embed_dim', type=int, default=96)
    ap.add_argument('--depths', type=int, nargs='+', default=[2,2])
    ap.add_argument('--heads', type=int, nargs='+', default=[3,6])
    ap.add_argument('--window_t', type=int, default=2)
    ap.add_argument('--window_h', type=int, default=8)
    ap.add_argument('--window_w', type=int, default=8)
    ap.add_argument('--patch_t', type=int, default=2)
    ap.add_argument('--patch_h', type=int, default=4)
    ap.add_argument('--patch_w', type=int, default=4)
    ap.add_argument('--adapter_ckpt', type=str, required=True)
    ap.add_argument('--n_latents', type=int, default=32)
    ap.add_argument('--adapter_layers', type=int, default=2)
    ap.add_argument('--adapter_heads', type=int, default=8)
    ap.add_argument('--files', type=str, nargs='+', required=True, help='ERA5 .nc files or globs (unquoted) or directories')
    ap.add_argument('--variables', type=str, nargs='+', required=True, help='Variables used in MAE pretrain (must match channels count)')
    ap.add_argument('--window_T', type=int, default=8)
    ap.add_argument('--window_H', type=int, default=64)
    ap.add_argument('--window_W', type=int, default=64)
    ap.add_argument('--stride_T', type=int, default=8)
    ap.add_argument('--stride_H', type=int, default=64)
    ap.add_argument('--stride_W', type=int, default=64)
    ap.add_argument('--date', type=str, required=True, help='Target date YYYY-MM-DD to fetch satellite window(s)')
    ap.add_argument('--anchor', type=str, default='last', choices=['first','middle','last'], help='Which timestep inside the window to anchor the date')
    ap.add_argument('--time_start', type=str, default=None)
    ap.add_argument('--time_end', type=str, default=None)
    ap.add_argument('--max_samples', type=int, default=4, help='Limit number of windows to decode for the date')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = args.device
    hrm, tokenizer = load_hrm_and_tokenizer(device)
    tok_emb, d_model = get_token_embedding_and_dim(hrm)

    # ---- Load ckpt to get shapes ----
    ckpt_raw = torch.load(args.mae_ckpt, map_location="cpu", weights_only=True)
    state = ckpt_raw.get("state_dict", ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw

    pew_key = None
    for k in state.keys():
        if k.endswith("encoder.patch_embed.proj.weight"):
            pew_key = k
            break
    if pew_key is None:
        raise KeyError("Could not find 'encoder.patch_embed.proj.weight' in MAE checkpoint.")
    w = state[pew_key]
    embed_dim_ckpt, in_chans_ckpt, p_t, p_h, p_w = w.shape

    mae = SatSwinMAE(
        in_chans=in_chans_ckpt,
        out_chans=in_chans_ckpt,
        embed_dim=embed_dim_ckpt,
        depths=tuple(args.depths),
        num_heads=tuple(args.heads),
        window_size=(args.window_t, args.window_h, args.window_w),
        patch_size=(p_t, p_h, p_w)
    ).to(device)
    mae.load_state_dict(state, strict=False)
    mae.eval()

    # infer d_vis from a dummy pass if needed (user should call adapter on real cubes)
    d_vis = args.embed_dim * (2 ** (len(args.depths)-1))
    adapter = VisionPrefixer(mae, d_vis=d_vis, d_model=d_model,
                             n_latents=args.n_latents, n_layers=args.adapter_layers, n_heads=args.adapter_heads).to(device)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device, weights_only=True), strict=False)
    adapter.eval()

    # -------- Build ERA5 dataset and select windows by date --------
    file_list = expand_input_files(args.files)
    if not file_list:
        raise SystemExit("[infer] No .nc files found after expanding --files. Example: --files ./dataset/raw_data/**/*.nc (no quotes)")
    window = {'T': args.window_T, 'H': args.window_H, 'W': args.window_W}
    stride = {'T': args.stride_T, 'H': args.stride_H, 'W': args.stride_W}
    era5 = ERA5CubeDataset(file_list, args.variables, window, stride,
                           time_start=args.time_start, time_end=args.time_end)

    time_name = _guess_time_name(era5.data)
    if time_name is None:
        raise SystemExit("[infer] Could not find a datetime coordinate in ERA5 dataset.")


    times = pd.to_datetime(era5.data[time_name].values)
    print(f"[infer] time_name={time_name}, count={len(times)}")
    print(f"[infer] first times: {times[:5]}")
    print(f"[infer] last  times: {times[-5:]}")

    # Peek computed window anchors for the first few windows
    anchors = []
    for i in range(min(32, len(era5.idxs))):  # era5.indices = list of (t0,h0,w0) or similar
        t0 = era5.idxs[i][0]
        t_end = t0 + (era5.window["T"] - 1)
        if args.anchor == "first":
            a = times[t0]
        elif args.anchor == "middle":
            a = times[t0 + (era5.window["T"] // 2)]
        else:  # last
            a = times[t_end]
        anchors.append(str(a))
    print("[infer] sample anchors:", anchors[:10])

    match_idxs = window_indices_for_date(era5, window, args.date, anchor=args.anchor)
    if not match_idxs:
        raise SystemExit(f"[infer] No windows found anchored on date {args.date}. Try a different --date/--anchor or adjust --time_start/--time_end.")
    if len(match_idxs) > args.max_samples:
        match_idxs = match_idxs[:args.max_samples]

    # -------- Generate text for each matched window --------
    tok_emb, _ = get_token_embedding_and_dim(hrm)
    for i, wi in enumerate(match_idxs, 1):
        cube, valid = era5[wi]  # (C,T,H,W), (T,H,W) mask
        cube = cube.unsqueeze(0).to(device)
        valid_b = valid.unsqueeze(0).to(device)

        # get visual prompts
        try:
            prompts = adapter(cube, valid_b)  # some versions accept both (cubes, valids)
        except TypeError:
            prompts = adapter(cube)           # fallback: only cubes

        # generate greedy
        text = generate_greedy(hrm, tok_emb, prompts, tokenizer, max_new_tokens=128, temperature=1.0)[0]
        print(f"\n=== Sample {i} / {len(match_idxs)} — Date {args.date} (idx {wi}) ===")
        print(text)
    print("\n[infer] Done.")

if __name__ == '__main__':
    main()
