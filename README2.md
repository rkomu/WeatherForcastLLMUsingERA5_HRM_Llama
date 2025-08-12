
# Weather HRM Pipeline (SatSwinMAE → Discrete Tokens → HRM)

This bundle gives you **full code** to:
1) Pretrain a from‑scratch **SatSwinMAE** on ERA5 cubes (self‑supervised),
2) Fit a **codebook** over encoder tokens (k‑means),
3) Convert ERA5 cubes into **HRM‑style discrete sequences** (JSONL) with a simple classification target (precipitation bins).

You can then fine‑tune your **HRM fork** on the generated dataset to predict weather (e.g., next‑hour precipitation bins).

## 0) Install
```bash
pip install torch xarray netCDF4 tqdm scikit-learn
```

## 1) Pretrain SatSwinMAE on ERA5
```bash
python -m sat_swin_mae.train_mae   --train_files /path/to/train/*.nc   --val_files   /path/to/val/*.nc   --variables u10 v10 t2m sp tp mwd mwp   --window_T 24 --window_H 64 --window_W 64   --stride_T 24 --stride_H 32 --stride_W 32   --batch_size 2 --epochs 20   --mask_ratio 0.75   --window_t 2 --window_h 8 --window_w 8   --patch_t 2 --patch_h 4 --patch_w 4   --out_dir checkpoints
```
This yields `checkpoints/satswinmae_epoch20.pt` (or similar).

## 2) Fit a codebook on encoder tokens
```bash
python tools/fit_codebook.py   --files /path/to/train/*.nc   --variables u10 v10 t2m sp tp mwd mwp   --mae_ckpt checkpoints/satswinmae_epoch20.pt   --k 4096   --out codebook_k4096.pkl
```

## 3) Build an HRM JSONL dataset (discrete tokens + precipitation bins)
```bash
python tools/build_hrm_dataset.py   --files /path/to/train/*.nc   --variables u10 v10 t2m sp mwd mwp   --target_var tp   --mae_ckpt checkpoints/satswinmae_epoch20.pt   --codebook codebook_k4096.pkl   --out_dir data_hrm_discrete   --tp_bins 0.0 0.2 1.0 5.0 10.0 20.0
```
This writes `data_hrm_discrete/era5_discrete.jsonl` with records like:
```json
{
  "inputs": [ 17, 845, 3,  ... ],
  "labels": [ 2 ],
  "puzzle_identifier": "era5_forecast_tp_bin",
  "grid": {"t": 12, "h": 16, "w": 16},
  "meta": {"file_index": 42, "target_scalar": 0.37}
}
```

> **Note:** For true *future* labels, modify the builder to read the **next window** (after the input window) for the `target_var`. The provided script uses the **last hour** of the input window as a simple proxy.

## 4) Fine‑tune HRM on the dataset
In your HRM fork:
- Point its dataset loader to read `era5_discrete.jsonl` (same shape as other puzzle datasets: a list of integer `inputs` + a single class label in `labels`).
- Keep the model **unchanged** (discrete tokens) and train for classification on `puzzle_identifier = "era5_forecast_tp_bin"`.

If you prefer **continuous regression** instead of bins, I can provide a second path that adds an input adapter and regression head to your HRM fork. Just say the word.
