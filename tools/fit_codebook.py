
import argparse, pickle, numpy as np, torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from sat_swin_mae.model import SatSwinMAE
from sat_swin_mae.dataset_era5 import ERA5CubeDataset

@torch.no_grad()
def encode_stream(mae, loader, device, max_tokens):
    collected = 0
    for cube in loader:
        z, _ = mae.encode_tokens(cube.to(device))  # (B,L,D)
        z = z.reshape(-1, z.shape[-1]).cpu().numpy()
        if max_tokens is not None and collected + len(z) > max_tokens:
            z = z[:max_tokens - collected]
        yield z
        collected += len(z)
        if max_tokens is not None and collected >= max_tokens:
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--window_T", type=int, default=24)
    ap.add_argument("--window_H", type=int, default=64)
    ap.add_argument("--window_W", type=int, default=64)
    ap.add_argument("--stride_T", type=int, default=24)
    ap.add_argument("--stride_H", type=int, default=64)
    ap.add_argument("--stride_W", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--mae_ckpt", type=str, required=True)
    ap.add_argument("--embed_dim", type=int, default=96)
    ap.add_argument("--depths", type=int, nargs="+", default=[2,2])
    ap.add_argument("--heads", type=int, nargs="+", default=[3,6])
    ap.add_argument("--window_t", type=int, default=2)
    ap.add_argument("--window_h", type=int, default=8)
    ap.add_argument("--window_w", type=int, default=8)
    ap.add_argument("--patch_t", type=int, default=2)
    ap.add_argument("--patch_h", type=int, default=4)
    ap.add_argument("--patch_w", type=int, default=4)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=2_000_000)
    ap.add_argument("--out", type=str, default="codebook_k4096.pkl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = {"T":args.window_T, "H":args.window_H, "W":args.window_W}
    stride = {"T":args.stride_T, "H":args.stride_H, "W":args.stride_W}
    ds = ERA5CubeDataset(args.files, args.variables, window, stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    in_chans = len(args.variables)
    mae = SatSwinMAE(
        in_chans=in_chans, out_chans=in_chans,
        embed_dim=args.embed_dim, depths=tuple(args.depths), num_heads=tuple(args.heads),
        window_size=(args.window_t, args.window_h, args.window_w),
        patch_size=(args.patch_t, args.patch_h, args.patch_w)
    ).to(device)
    mae.load_state_dict(torch.load(args.mae_ckpt, map_location=device))
    mae.eval()

    km = MiniBatchKMeans(n_clusters=args.k, batch_size=10_000, verbose=1)
    for z in encode_stream(mae, loader, device, args.max_tokens):
        km.partial_fit(z)
    with open(args.out, "wb") as f:
        pickle.dump(km, f)
    print(f"Saved codebook to {args.out}")

if __name__ == "__main__":
    main()
