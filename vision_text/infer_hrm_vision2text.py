
import argparse, os, torch
from sat_swin_mae.model import SatSwinMAE
from vision_text.vision_adapter import VisionPrefixer
from transformers import AutoTokenizer
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.hrm.hrm_lm_adapter import HRMAsTextLM

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

        "forward_dtype": "float32",        # safer default on consumer GPUs
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
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device), strict=False)
    adapter.eval()

    print("Adapter and HRM loaded. To generate: prompts = adapter(cubes); texts = generate_greedy(hrm, tok_emb, prompts, tokenizer)")

if __name__ == '__main__':
    main()
