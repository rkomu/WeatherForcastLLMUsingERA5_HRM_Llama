# vision_text/infer_tinyllama_vision2text.py
# Inference: ERA5 window -> SatSwinMAE tokens -> VisionPrefixer soft prompts -> TinyLlama.generate()

import os
import glob
import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sat_swin_mae.model import SatSwinMAE
from sat_swin_mae.dataset_era5 import ERA5CubeDataset
from vision_text.vision_adapter import VisionPrefixer


def expand_files(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        hits = glob.glob(p)
        out.extend(hits if hits else [p])
    return sorted(list(dict.fromkeys(out)))


@torch.no_grad()
def build_soft_prompts(mae: SatSwinMAE, prefixer: VisionPrefixer, cube: torch.Tensor):
    vis_tokens, _ = mae.encode_tokens(cube)     # (1, L_vis, d_vis)
    soft_prompts = prefixer(vis_tokens)         # (1, n_latents, d_model)
    return soft_prompts


@torch.no_grad()
def generate_with_soft_prompts(
    lm: AutoModelForCausalLM,
    tok: AutoTokenizer,
    soft_prompts: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
):
    device = soft_prompts.device
    emb = lm.get_input_embeddings()

    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    bos_ids = torch.tensor([[bos_id]], device=device)
    bos_emb = emb(bos_ids)                                 # (1,1,d_model)

    inputs_embeds = torch.cat([soft_prompts, bos_emb], 1)  # (1, n_latents+1, d_model)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    gen_ids = lm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0 or top_k > 0 or top_p < 1.0),
        temperature=max(1e-6, temperature),
        top_p=top_p if top_p > 0 else 1.0,
        top_k=top_k if top_k > 0 else 0,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    # When using inputs_embeds, HF returns only newly generated token ids
    text = tok.decode(gen_ids[0], skip_special_tokens=True)
    return text


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--window_T", type=int, default=2)
    ap.add_argument("--window_H", type=int, default=64)
    ap.add_argument("--window_W", type=int, default=64)
    ap.add_argument("--stride_T", type=int, default=1)
    ap.add_argument("--stride_H", type=int, default=64)
    ap.add_argument("--stride_W", type=int, default=64)
    ap.add_argument("--time_start", type=str, default=None)
    ap.add_argument("--time_end", type=str, default=None)

    ap.add_argument("--max_samples", type=int, default=4)
    ap.add_argument("--anchor", type=str, default="first", choices=["first", "middle", "last"])

    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama_v1.1")
    ap.add_argument("--n_latents", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)

    ap.add_argument("--mae_ckpt", type=str, required=True)
    ap.add_argument("--adapter_ckpt", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main():
    args = build_argparser().parse_args()

    files = expand_files(args.files)
    window = {"T": args.window_T, "H": args.window_H, "W": args.window_W}
    stride = {"T": args.stride_T, "H": args.stride_H, "W": args.stride_W}

    dset = ERA5CubeDataset(
        files, args.variables, window, stride,
        time_start=args.time_start, time_end=args.time_end
    )
    in_chans = int(dset.C)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    ).to(args.device).eval()

    # MAE (frozen)
    mae = SatSwinMAE(
        in_chans=in_chans, out_chans=in_chans,
        embed_dim=96, depths=(2, 2), num_heads=(3, 6),
        window_size=(2, 7, 7), patch_size=(2, 4, 4), mask_ratio=0.75
    )
    mae.load_state_dict(torch.load(args.mae_ckpt, map_location="cpu"), strict=False)
    mae.to(args.device).eval()

    # VisionPrefixer (load trained)
    d_vis = mae.encoder.stages[-1][-1].attn.qkv.in_features
    d_model = int(lm.config.hidden_size)
    prefixer = VisionPrefixer(
        d_vis=d_vis, d_model=d_model,
        n_latents=args.n_latents, n_layers=2, n_heads=8, dropout=0.0
    ).to(args.device)
    prefixer.load_state_dict(torch.load(args.adapter_ckpt, map_location="cpu"))
    prefixer.eval()

    # pick sample windows
    N = len(dset)
    if N == 0:
        raise SystemExit("No windows matched the configuration.")
    if args.anchor == "first":
        step = max(1, N // args.max_samples)
        idxs = [i for i in range(0, min(N, args.max_samples * step), step)][:args.max_samples]
    elif args.anchor == "middle":
        mid = N // 2
        start = max(0, mid - args.max_samples // 2)
        idxs = list(range(start, min(N, start + args.max_samples)))
    else:
        start = max(0, N - args.max_samples)
        idxs = list(range(start, N))

    for i in idxs:
        cube, valid = dset[i]
        cube = cube.unsqueeze(0).to(args.device)
        soft = build_soft_prompts(mae, prefixer, cube)
        caption = generate_with_soft_prompts(
            lm, tok, soft,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k
        )
        print(f"[sample {i:05d}] {caption}")


if __name__ == "__main__":
    main()
    