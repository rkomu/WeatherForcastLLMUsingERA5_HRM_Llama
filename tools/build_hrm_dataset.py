
import os, argparse, json, pickle, numpy as np, torch
from torch.utils.data import DataLoader
from sat_swin_mae.model import SatSwinMAE
from sat_swin_mae.dataset_era5 import ERA5CubeDataset

def bin_scalar(x, edges):
    # edges: list of right-closed bin edges (e.g., [0,1,2,5,10,20,50])
    # returns bin index in [0, len(edges)]
    for i, e in enumerate(edges):
        if x <= e: return i
    return len(edges)

@torch.no_grad()
def encode_ids(mae, km, cube):
    z, grid = mae.encode_tokens(cube.unsqueeze(0))  # (1,L,D)
    z = z[0].cpu().numpy()
    ids = km.predict(z)  # (L,)
    return ids.tolist(), grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--target_var", type=str, default="tp")   # total precipitation
    ap.add_argument("--future_hours", type=int, default=1)    # label: next-N hour mean
    ap.add_argument("--window_T", type=int, default=24)
    ap.add_argument("--window_H", type=int, default=64)
    ap.add_argument("--window_W", type=int, default=64)
    ap.add_argument("--stride_T", type=int, default=24)
    ap.add_argument("--stride_H", type=int, default=64)
    ap.add_argument("--stride_W", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--mae_ckpt", type=str, required=True)
    ap.add_argument("--codebook", type=str, required=True)
    ap.add_argument("--embed_dim", type=int, default=96)
    ap.add_argument("--depths", type=int, nargs="+", default=[2,2])
    ap.add_argument("--heads", type=int, nargs="+", default=[3,6])
    ap.add_argument("--window_t", type=int, default=2)
    ap.add_argument("--window_h", type=int, default=8)
    ap.add_argument("--window_w", type=int, default=8)
    ap.add_argument("--patch_t", type=int, default=2)
    ap.add_argument("--patch_h", type=int, default=4)
    ap.add_argument("--patch_w", type=int, default=4)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tp_bins", type=float, nargs="+", default=[0.0, 0.2, 1.0, 5.0, 10.0, 20.0])  # mm/hr
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = {"T":args.window_T, "H":args.window_H, "W":args.window_W}
    stride = {"T":args.stride_T, "H":args.stride_H, "W":args.stride_W}

    # Make dataset with target var included (for labels)
    variables = list(args.variables)
    if args.target_var not in variables:
        variables.append(args.target_var)
    ds = ERA5CubeDataset(args.files, variables, window, stride)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    in_chans = len(variables)
    mae = SatSwinMAE(
        in_chans=in_chans, out_chans=in_chans,
        embed_dim=args.embed_dim, depths=tuple(args.depths), num_heads=tuple(args.heads),
        window_size=(args.window_t, args.window_h, args.window_w),
        patch_size=(args.patch_t, args.patch_h, args.patch_w)
    ).to(device)
    mae.load_state_dict(torch.load(args.mae_ckpt, map_location=device))
    mae.eval()

    with open(args.codebook, "rb") as f:
        km = pickle.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    shard_path = os.path.join(args.out_dir, "era5_discrete.jsonl")
    with open(shard_path, "w") as f:
        for i, cube in enumerate(loader):
            cube = cube.to(device)[0]  # (C,T,H,W)
            # separate target var index
            tvar_idx = variables.index(args.target_var)
            past = cube            # we use the past window as inputs
            # label: mean of next future_hours for target var — you'll need to fetch next windows in your pipeline.
            # Here, as a placeholder, we use the LAST hour in the window as a proxy label.
            target_map = past[tvar_idx, -1]  # (H,W)
            y_scalar = float(target_map.mean().item())
            y_bin = bin_scalar(y_scalar, args.tp_bins)
            # ids from encoder
            ids, grid = encode_ids(mae, km, past)
            rec = {
                "inputs": ids,
                "labels": [int(y_bin)],
                "puzzle_identifier": "era5_forecast_tp_bin",
                "grid": {"t": int(grid[0]), "h": int(grid[1]), "w": int(grid[2])},
                "meta": {"file_index": i, "target_scalar": y_scalar}
            }
            f.write(json.dumps(rec)+"
")
    print(f"Wrote HRM dataset shard: {shard_path}")
    print("NOTE: For true future labels, build sequences with an explicit 'future' window aligned after the input window.")
if __name__ == "__main__":
    main()
