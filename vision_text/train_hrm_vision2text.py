
import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sat_swin_mae.model import SatSwinMAE
from vision_text.vision_adapter import VisionPrefixer
from vision_text.dataset_sat_text import ERA5CaptionDatasetCSV
from transformers import AutoTokenizer
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.hrm.hrm_lm_adapter import HRMAsTextLM
from functools import partial

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--variables', nargs='+', required=True)
    ap.add_argument('--window_T', type=int, default=8)
    ap.add_argument('--window_H', type=int, default=64)
    ap.add_argument('--window_W', type=int, default=64)
    ap.add_argument('--stride_T', type=int, default=8)
    ap.add_argument('--stride_H', type=int, default=64)
    ap.add_argument('--stride_W', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
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
    ap.add_argument('--n_latents', type=int, default=32)
    ap.add_argument('--adapter_layers', type=int, default=2)
    ap.add_argument('--adapter_heads', type=int, default=8)
    ap.add_argument('--time_start', type=str, default=None)
    ap.add_argument('--time_end', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', type=str, default='checkpoints_v2t')
    ap.add_argument('--caption_csv', type=str, default=None, help="CSV with 'date' and 'event description' columns", required=True)
    ap.add_argument('--drop_if_no_caption', action='store_true', help="Drop windows with no caption for their date")
    ap.add_argument('--anchor', type=str, default='last', choices=['first','middle','last'], help="Which timestep to use to compute the date")
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()

def collate_with_pad(batch, pad_id: int):
    cubes, valids, input_ids = zip(*batch)            # lists
    import torch
    cubes = torch.stack(cubes, 0)
    valids = torch.stack(valids, 0)
    lens = [len(x) for x in input_ids]
    max_len = max(lens)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded[i, :len(ids)] = ids
    attn = (padded != pad_id).long()
    return cubes, valids, padded, attn

def main():
    args = parse_args()
    device = args.device

    # HRM + tokenizer
    hrm, tokenizer = load_hrm_and_tokenizer(device)
    tok_emb, d_model = get_token_embedding_and_dim(hrm)

    # ---- Load MAE CKPT METADATA FIRST (to match in_chans & patch) ----
    ckpt_raw = torch.load(args.mae_ckpt, map_location="cpu", weights_only=True)
    state = ckpt_raw.get("state_dict", ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw

    # robustly find the patch-embed weight key
    pew_key = None
    for k in state.keys():
        if k.endswith("encoder.patch_embed.proj.weight"):
            pew_key = k
            break
    if pew_key is None:
        raise KeyError("Could not find 'encoder.patch_embed.proj.weight' in MAE checkpoint.")

    w = state[pew_key]  # shape: [embed_dim, in_chans, p_t, p_h, p_w]
    embed_dim_ckpt, in_chans_ckpt, p_t, p_h, p_w = w.shape

    # ---- Dataset / Loader (build BEFORE MAE so we can sanity-check channels) ----
    # Dataset / Loader
    window = {'T': args.window_T, 'H': args.window_H, 'W': args.window_W}
    stride = {'T': args.stride_T, 'H': args.stride_H, 'W': args.stride_W}
    ds = ERA5CaptionDatasetCSV(
        args.files, args.variables, window, stride,
        tokenizer=tokenizer,
        csv_path=args.caption_csv,
        time_start=args.time_start, time_end=args.time_end,
        max_len=128,
        drop_if_no_caption=args.drop_if_no_caption,
        anchor=args.anchor,
    )
    pad_id = (getattr(tokenizer, 'pad_token_id', None)
            if getattr(tokenizer, 'pad_token_id', None) is not None
            else getattr(tokenizer, 'eos_token_id', 0))
    collate_fn = partial(collate_with_pad, pad_id=pad_id)    
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_fn)

    # sanity: dataset channels must match ckpt channels
    dataset_C = int(ds.inner.C)
    if dataset_C != in_chans_ckpt:
        raise SystemExit(
            f"[vision2text] Channel mismatch: dataset C={dataset_C} vs MAE ckpt in_chans={in_chans_ckpt}. "
            "Your MAE was pretrained with pressure-level stacking (e.g., 17). "
            "Make sure ERA5CubeDataset returns the same C (pressure_handling/levels) when training the adapter."
        )

    # ---- Build MAE with CKPT-MATCHING SHAPES ----
    mae = SatSwinMAE(
        in_chans=in_chans_ckpt,
        out_chans=in_chans_ckpt,
        embed_dim=embed_dim_ckpt,                      # match ckpt
        depths=tuple(args.depths),
        num_heads=tuple(args.heads),
        window_size=(args.window_t, args.window_h, args.window_w),
        patch_size=(p_t, p_h, p_w),                    # match ckpt
        mask_ratio=0.75
    ).to(device)

    # load weights (now shapes match)
    mae.load_state_dict(state, strict=False)
    mae.eval()
    for p in mae.parameters():
        p.requires_grad = False

    # Probe d_vis
    cube0, _, _ = ds[0]
    with torch.no_grad():
        z0, _ = mae.encode_tokens(cube0.unsqueeze(0).to(device))
    d_vis = z0.shape[-1]

    # Vision→Text adapter
    from vision_text.vision_adapter import VisionPrefixer
    adapter = VisionPrefixer(mae, d_vis=d_vis, d_model=d_model,
                             n_latents=args.n_latents, n_layers=args.adapter_layers, n_heads=args.adapter_heads).to(device)

    # Freeze HRM (BLIP-2 style); train only adapter
    for p in hrm.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        adapter.train()
        total = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        for cubes, valids, input_ids, text_attn in pbar:
            cubes = cubes.to(device)
            input_ids = input_ids.to(device)
            text_attn = text_attn.to(device)

            prompts = adapter(cubes)                  # (B,M,d_model)
            txt_emb = tok_emb(input_ids)             # (B,L,d_model)
            inputs_embeds = torch.cat([prompts, txt_emb], dim=1)
            B, M, _ = prompts.shape
            prompt_attn = torch.ones((B, M), device=device, dtype=text_attn.dtype)
            attn = torch.cat([prompt_attn, text_attn], dim=1)

            ignore = -100
            labels = torch.full_like(input_ids, fill_value=ignore)
            labels_full = torch.full((B, M + input_ids.size(1)), ignore, device=device, dtype=torch.long)
            
            # Supervise only *non-pad* text tokens; ignore prompts and pad.
            ignore = -100
            B, M, _ = prompts.shape
            labels_full = torch.full((B, M + input_ids.size(1)), ignore, device=device, dtype=torch.long)

            # mask where text tokens are valid (1) vs pad (0)
            text_mask = text_attn.bool()
            labels_text = input_ids.masked_fill(~text_mask, ignore)
            labels_full[:, M:] = labels_text

            out = forward_with_embeds(hrm, inputs_embeds, attention_mask=attn)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels_full.view(-1),
                ignore_index=ignore
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()

            total += loss.item() * B
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        torch.save(adapter.state_dict(), os.path.join(args.out_dir, f"adapter_epoch{epoch}.pt"))
        print(f"Saved adapter_epoch{epoch}.pt")

if __name__ == '__main__':
    main()
