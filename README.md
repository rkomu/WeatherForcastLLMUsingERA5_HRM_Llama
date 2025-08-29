
# WeatherLM: Vision-to-Text Weather Narratives from ERA5 usi**High-level answer:** Freeze a strong **3D masked autoencoder** for vision (### 6.1 Wh### 6.2 Why "soft prompts" (pr- **Inductive bias**: the heavy lifting (spatiotemporal abstraction) is alre## 12) Diagnostics & Tips

- If loss ≈ 10.8 and flat: verify labels ignore prompts, adapter params in the optimizer, no `.detach()` on prompts, no `no_grad` wrapping the language model forward with embeds.  
- If "No windows": widen time or lower `window_T` (≥ `patch_t`).  
- Channel mismatch vs ckpt: ensure dataset variables/levels sum to the encoder's expected `in_chans`.  
- Keep `LM.seq_len ≥ M + max_caption_len`.ne by SatSwinMAE. The adapter's job is **affine re-embedding** with mild nonlinearity into the language model's space.
- **Latency/VRAM**: smaller = faster and leaves budget for longer contexts if needed.x) and not cross-attention?
- **Parameter & data efficiency**: a prefixer adds **tiny** parameter count; cross-attn adds full Q/K/V projections and blocks that are harder to train with limited captions.
- **Architectural simplicity**: no surgery inside the language model; works with *any* LM that exposes `inputs_embeds`.
- **Stability**: prefix-tuning is well-behaved with frozen LMs; gradients flow through the LM to the **inputs**, teaching the adapter alignment "vision → words" without destabilizing the LM.
- **Sequence control**: the number of soft tokens **M** is explicit; we can budget the LM's `seq_len` (`seq_len ≥ M + max_caption_len`). does
- Input: `z ∈ ℝ^{B×M×Dv}` (M selected/pooled vision tokens).
- MLP: `Dv → dH` with **SiLU or GELU** activations, typically 2–4 layers, optional residual/LayerScale.
- Output: `P ∈ ℝ^{B×M×dH}` — **soft prompt tokens** in the **same space** as the language model's word embeddings.
- Concatenate with text embeddings: `X = [P | E_y]`, then run the language model with `inputs_embeds`.inMAE) and a reasoning-oriented **language model** (HRM-ACTv1 or TinyLlama-1.1B). Train only a small **Vision→Text adapter** that maps visual latents to **soft text prompts** consumed by the language model. This preserves each backbone's strength and avoids catastrophic forgetting.g SatSwinMAE + HRM/TinyLlama

> ****Why HRM (vs a generic LM)**: weather narratives often benefit from **iterative** reasoning ("if low deepens then…"). HRM's H/L cycles and ACT provide a natural mechanism for this, even when frozen—the adapter learns to place prompts that guide these steps.

---

## 5.1) TinyLlama-1.1B (Alternative Language Backbone) — What & Why

As an alternative to HRM-ACTv1, we also support **TinyLlama-1.1B** as the language backbone, particularly useful for faster experimentation and resource-constrained environments.

### 5.1.1 Architecture
- **Base model**: TinyLlama/TinyLlama_v1.1 (1.1B parameters)
- **Architecture**: Transformer decoder with **RMSNorm**, **SwiGLU** MLPs, **RoPE** positional encoding
- **Vocabulary**: ~32K tokens optimized for general text generation
- **Context length**: 2048 tokens (sufficient for weather captions)

### 5.1.2 Integration with Vision
- **Frozen backbone**: TinyLlama weights remain frozen during vision-text training
- **Soft prompt injection**: Vision adapter outputs are prepended as `inputs_embeds` to the language model
- **QLoRA support**: Optional 4-bit quantization with LoRA adapters for memory efficiency

### 5.1.3 Training Efficiency Features
- **QLoRA (Quantized LoRA)**: 4-bit quantization reduces memory footprint by ~75%
- **LoRA adapters**: Low-rank adaptation on attention and MLP layers (typical: `r=16`, `α=32`)
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Gradient checkpointing**: Reduces memory usage during backpropagation

### 5.1.4 Why TinyLlama (vs HRM)?
- **Faster iteration**: Smaller model = faster training and inference for prototyping
- **Lower resource requirements**: ~1.1B vs larger HRM variants; works on single consumer GPUs
- **Pretrained language priors**: Benefits from extensive text pretraining on diverse corpora
- **Production ready**: Well-tested transformer architecture with robust tokenization
- **Compatibility**: Standard HuggingFace interface for easy integration with existing tools

### 5.1.5 When to use TinyLlama vs HRM
- **TinyLlama**: 
  - Rapid prototyping and experimentation
  - Resource-constrained environments (single GPU, limited VRAM)
  - Baseline comparisons and ablation studies
  - Production deployments prioritizing speed over sophisticated reasoning
- **HRM**: 
  - Complex weather reasoning requiring iterative refinement
  - Applications needing adaptive computation and variable inference steps
  - Research into hierarchical reasoning mechanisms

---

## 6) Vision→Text Adapter (Prefixer) — What, How, and **Why**** — This README explains the **technical design and rationale** for the model. It focuses on *what the pieces are*, *how they fit*, and *why we chose them*. (Setup/installation is intentionally omitted here.)

---

## 1) Problem Statement & Design Goals

We want a model that **looks at spatiotemporal meteorological fields (ERA5)** and **writes human-readable weather narratives**. The design must:

- Respect the **3D structure** (channels × time × lat × lon).
- Be **data efficient**: captioned weather text is scarce.
- Leverage abundant **unlabeled** data to learn physical structure.
- Produce **coherent, causal text** (e.g., “a trough deepens, showers spread east”).

**High-level answer:** Freeze a strong **3D masked autoencoder** for vision (SatSwinMAE) and a reasoning-oriented **language model** (HRM-ACTv1). Train only a small **Vision→Text adapter** that maps visual latents to **soft text prompts** consumed by HRM. This preserves each backbone’s strength and avoids catastrophic forgetting.

---

## 2) System Overview

```
ERA5 windows (C × T × H × W)
          │
    SatSwinMAE encoder (frozen) ──► z: B × M× Dv   (visual latents)
          │
 Vision→Text Adapter (trainable) ──► P: B × M× dH  (soft prompts in LM space)
          │
      Language Model (frozen) ─────────► logits over vocab → text
      [HRM-ACTv1 or TinyLlama-1.1B]
```

- **SatSwinMAE** — 3D Swin Transformer + MAE pretraining to learn generic spatiotemporal structure.
- **Adapter** — small MLP mapping vision latent dim **Dv** → LM embed dim **dH**; outputs **M** soft prefix tokens.
- **Language Model** — frozen LM (HRM-ACTv1 or TinyLlama-1.1B); consumes `inputs_embeds` (prefix + word embeddings) and emits token logits.

**Why this separation?**  
Self-supervised **vision pretraining** scales cheaply; **LM priors** handle discourse. **Adapter-only** training is stable and sample-efficient.

---

## 3) ERA5 Data → Windows

- Variables (e.g., `u10`, `v10`, `t`, `r`, `sp`, `tp`, `ssrd`, ...). Pressure-level fields are stacked as channels → total **C** must match the SatSwinMAE checkpoint’s `in_chans`.
- **Uppercase window** sizes: `window_T × window_H × window_W` (e.g., `8×64×64`).
- **Uppercase stride**: `stride_T/H/W` controls overlap (e.g., `4/32/32` = 50% overlap).  
- Time filtering: `--time_start/--time_end` trims the dataset; still need ≥`window_T` timesteps to form at least one clip.
- **Anchor** (`first|middle|last`): defines which timestep’s date is used to join captions and select inference windows.

**Spatial tiling example** (0.25° grid `721×1440`, window `64`, stride `32`):
- `nH = floor((721-64)/32)+1 = 21`, `nW = floor((1440-64)/32)+1 = 44` → **924** crops per timestep.

---

## 4) SatSwinMAE (Vision Backbone) — What & Why

### 4.1 Patch Embedding (Tokenization)
- Conv3D with kernel=stride = `(patch_t, patch_h, patch_w)` (e.g., `2,4,4`) partitions the cube into **non-overlapping 3D patches**.
- Token grid:
  ```
  T' = floor(T/patch_t), H' = floor(H/patch_h), W' = floor(W/patch_w)
  tokens = T'·H'·W'
  ```
- **Why**: 3D patches respect temporal continuity and reduce sequence length in a physically sensible way. Choose `window_*` as multiples of `patch_*` to avoid edge drops.

### 4.2 Swin Transformer (3D)
- **Local attention** on token windows with **shifts** to connect neighborhoods.
- **Lowercase window** sizes (in token units), e.g., `2×8×8`, divide the token grid for efficient attention.
- **Why**: Mesoscale structure is locally coherent (fronts, jet streaks); windowed attention is **O(n·w²)** instead of **O(n²)** and empirically strong for geospatial data.

### 4.3 Masked Autoencoding (MAE)
- Randomly mask ~75% patches; reconstruct with a light decoder.
- **Why**: Uses abundant unlabeled ERA5 to learn robust spatiotemporal features, capturing synoptic patterns without captions.

### 4.4 For V2T
- We freeze the encoder and extract **visual latents** `z ∈ ℝ^{B×(T'·H'·W')×Dv}`. We then **select or pool** latents to **M** tokens before the adapter (see §6).

---

## 5) HRM-ACTv1 (Language Backbone) — What & Why

- **LM core**: embeddings → Transformer blocks with **RoPE**, **RMSNorm**, **SwiGLU** MLPs; **PyTorch SDPA** attention (portable & stable in fp32/bf16/fp16).
- **Hierarchical reasoning**: two interacting levels (**H/L**) with configurable cycles; mimics coarse→fine iterative refinement.
- **ACT (Adaptive Computation Time)**: halting head allows variable steps per example.
- **Inputs-Embeds path**: `forward_with_embeds(inputs_embeds, attention_mask)` so the first **M** positions can be **soft prompts** from vision.

**Why HRM (vs a generic LM)**: weather narratives often benefit from **iterative** reasoning (“if low deepens then…”). HRM’s H/L cycles and ACT provide a natural mechanism for this, even when frozen—the adapter learns to place prompts that guide these steps.

---

## 6) Vision→Text Adapter (Prefixer) — What, How, and **Why**

### 6.1 What it does
- Input: `z ∈ ℝ^{B×M×Dv}` (M selected/pooled vision tokens).
- MLP: `Dv → dH` with **SiLU or GELU** activations, typically 2–4 layers, optional residual/LayerScale.
- Output: `P ∈ ℝ^{B×M×dH}` — **soft prompt tokens** in the **same space** as HRM’s word embeddings.
- Concatenate with text embeddings: `X = [P | E_y]`, then run HRM with `inputs_embeds`.

### 6.2 Why “soft prompts” (prefix) and not cross-attention?
- **Parameter & data efficiency**: a prefixer adds **tiny** parameter count; cross-attn adds full Q/K/V projections and blocks that are harder to train with limited captions.
- **Architectural simplicity**: no surgery inside HRM; works with *any* LM that exposes `inputs_embeds`.
- **Stability**: prefix-tuning is well-behaved with frozen LMs; gradients flow through the LM to the **inputs**, teaching the adapter alignment “vision → words” without destabilizing the LM.
- **Sequence control**: the number of soft tokens **M** is explicit; we can budget HRM’s `seq_len` (`seq_len ≥ M + max_caption_len`).

### 6.3 Why **SiLU/GELU** in the adapter MLP?
We choose **smooth, non-linear activations** that are known to work well in Transformer MLPs and mixed-precision training.

- **SiLU (Swish‑1)**: `x · sigmoid(x)`  
  - **Smooth** and **non-monotonic** near 0 → richer gating behavior than ReLU.  
  - **No hard zero** region → avoids “dead neurons”, keeps gradient flowing for small negatives.  
  - Empirically strong in vision backbones and adapters; cheap to compute and AMP/bf16 friendly.
- **GELU**: approximates input-dependent gating with a Gaussian CDF weighting.  
  - Standard in BERT/ViT MLPs; similarly smooth; excellent with LayerNorm/RMSNorm statistics.  
  - Slightly different curvature than SiLU; both work nearly interchangeably in practice.

**Why not ReLU/tanh?**
- **ReLU**: sparse gradients for negatives; adapters are small and must learn subtle alignments — losing gradient in half the domain hurts.  
- **tanh**: bounded outputs saturate early; we want the adapter to reach the full **dH** dynamic range used by HRM’s token embeddings.

**Practical rule**:  
- Default to **SiLU** in the adapter (great gradient flow, lightweight).  
- If you want to match HRM/ViT conventions exactly, use **GELU** — results are typically on par.

### 6.4 Why an MLP (and not a heavier transformer) for the adapter?
- **Inductive bias**: the heavy lifting (spatiotemporal abstraction) is already done by SatSwinMAE. The adapter’s job is **affine re-embedding** with mild nonlinearity into HRM’s space.  
- **Overfitting risk**: with limited captions, a large adapter would overfit quickly.  
- **Latency/VRAM**: smaller = faster and leaves budget for longer contexts if needed.

### 6.5 How we pick **M** (number of prompt tokens)
- Start with **M = 32**; try 16/64 in ablations.  
- Ensure `LM.seq_len ≥ M + max_caption_len`.  
- Increasing **M** helps detail recall up to a point; too large → longer sequences with diminishing returns.

### 6.6 Pooling/selection from the vision grid to M tokens — Why & options
We often have many encoder tokens (`T'·H'·W'`). We reduce to **M** to keep HRM input compact:

- **Mean pooling** per local block or global — **robust** and parameter-free; good baseline.  
- **Attention pooling** with a few learned queries — focuses on salient synoptic structures.  
- **Strided subsampling** — simplest path if token grid is already dense.

**Why reduce?** Language model sequence budget is precious. Pooling summarizes redundant local patterns (e.g., broad stratiform cloud decks) without losing essential synoptic cues (lows, fronts, jets).

---

## 7) Training: What Learns & Why It Works

- **Frozen**: SatSwinMAE, Language Model weights.  
- **Trainable**: the adapter MLP (and optionally pooling params).

**Mechanism**: next-token **cross-entropy** on caption tokens. The loss backpropagates **through the frozen language model** to the **adapter outputs** (prompts). The adapter learns to place the right vectors so the LM emits the desired words.

**Loss baseline**: with a GPT‑2 tokenizer `V≈50k`, a uniform-guess CE ≈ `ln(V) ≈ 10.8` nats. Early loss around 10–11 is normal and quickly drops if the wiring is correct.

**When to unfreeze**: if generations stay generic, unfreeze the **last LM block** (or add **LoRA** for TinyLlama) with a tiny LR (e.g., `1e‑5`) to better bind phrasing to visual cues.

---

## 8) Inference by Date — What & Why

- **What**: choose windows whose **anchor timestep** matches `--date`. For each, extract latents, map to prompts, decode text.  
- **Why**: date-driven evaluation reflects how forecasters summarize a day’s synoptic state; it also enables easy qualitative checks against known events.

Key controls: `--date`, `--anchor`, `--time_start/--time_end`, `--window_*`, `--stride_*`, `--variables`, `--max_samples`.

---

## 9) Shapes & Math (Concise)

- Cube `x ∈ ℝ^{B×C×T×H×W}`.  
- Patch `(p_t,p_h,p_w)` → token grid `T'×H'×W'`.  
- Encoder yields `z_grid ∈ ℝ^{B×(T'·H'·W')×Dv}` → pooled to `z ∈ ℝ^{B×M×Dv}`.  
- Adapter `A: ℝ^{Dv}→ℝ^{dH}` → `P = A(z) ∈ ℝ^{B×M×dH}`.  
- Text embeddings `E_y ∈ ℝ^{B×L×dH}`. Inputs: `X = [P | E_y]`.  
- Language Model `F(X) → logits ∈ ℝ^{B×(M+L)×V}`.  
- Labels ignore prompts/pad (`-100`). Loss `CE(logits, labels)` updates **A** only.

---

## 10) Most Impactful Hyperparameters (and why)

- **window_T/H/W vs patch_t/h/w**: controls **token count**; choose multiples to avoid edge losses.  
- **Swin window_t/h/w**: wider windows capture broader context but raise attention cost per block.  
- **M (prompt length)**: larger M carries more scene detail until it hits sequence limits.  
- **Adapter activations (SiLU/GELU)**: smooth gradients, strong empirical performance in transformer MLPs, mixed‑precision friendly — critical for a **small** network tasked with fine alignment.  
- **LR for adapter**: `1e‑4`–`5e‑4` typically; small warmup helps stability.  
- **Precision**: start fp32 (debug), switch to bf16 for speed once stable.

---

## 11) Design Choices — Alternatives Considered (and why we didn’t pick them)

- **Cross-attention fusion** (vision↔text): more expressive, but heavier, harder to stabilize with little text; more parameters and VRAM. Prefixer achieves most of the gain at a fraction of complexity.  
- **Training HRM end‑to‑end**: risks forgetting general language competence; requires much more text; slower experiments.  
- **ReLU/tanh in adapter**: poorer gradient flow / saturation; empirically worse for tiny adapters aligning two high‑dimensional manifolds.  
- **FlashAttention**: great speed but fragile Python/Torch/CUDA build matrix; SDPA is portable and plenty fast for our sequence lengths.

---

## 12) Diagnostics & Tips

- If loss ≈ 10.8 and flat: verify labels ignore prompts, adapter params in the optimizer, no `.detach()` on prompts, no `no_grad` wrapping the HRM forward with embeds.  
- If “No windows”: widen time or lower `window_T` (≥ `patch_t`).  
- Channel mismatch vs ckpt: ensure dataset variables/levels sum to the encoder’s expected `in_chans`.  
- Keep `HRM.seq_len ≥ M + max_caption_len`.

---

## 13) Minimal Training Pseudocode

```python
with torch.no_grad():
    z = mae.encode_tokens(cubes)            # (B, M, Dv)

prompts = adapter(z)                        # (B, M, dH)
E_y = lm.embed_tokens(input_ids)            # (B, L, dH)

X = torch.cat([prompts, E_y], dim=1)        # (B, M+L, dH)
mask = torch.cat([torch.ones(B, M, device=X.device), text_attn], dim=1)

logits = lm.forward_with_embeds(X, attention_mask=mask)  # (B, M+L, V)

labels = torch.full((B, M+L), -100, device=X.device, dtype=torch.long)
labels[:, M:] = input_ids.masked_fill(~text_attn.bool(), -100)

loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
loss.backward(); opt.step()
```

---

## 14) File Layout (orientation)

- `sat_swin_mae/` — encoder, patch/embed, MAE training.  
- `models/hrm/` — HRM-ACTv1 (hierarchical LM), layers (SDPA attention).  
- `vision_text/` — adapter training and date‑driven inference (both HRM and TinyLlama versions).
  - `train_hrm_vision2text.py` — HRM-based vision-to-text training
  - `train_tinyllama_vision2text.py` — TinyLlama-based vision-to-text training with QLoRA support
  - `infer_tinyllama_vision2text.py` — TinyLlama inference script
- `dataset_*` — ERA5 & caption datasets (multi‑caption per date).
- `checkpoints_v2t/` — trained adapter checkpoints and LoRA adapters.

---

## 15) Training Scripts & Usage

### 15.1 TinyLlama Vision-to-Text Training

Basic training with TinyLlama:
```bash
python -m vision_text.train_tinyllama_vision2text \
  --files "dataset/raw_data/nc_*/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --window_T 8 --window_H 64 --window_W 64 \
  --stride_T 4 --stride_H 32 --stride_W 32 \
  --mae_ckpt checkpoints/satswinmae_epoch59.pt \
  --caption_csv dataset/weather/weather_jan_may_2024.csv \
  --caption_date_col date --caption_text_col "event description" \
  --drop_if_no_caption --anchor last \
  --batch_size 32 --epochs 100 --lr 0.0001 \
  --n_latents 32 --adapter_layers 2 --adapter_heads 8
```

With QLoRA (memory efficient):
```bash
python -m vision_text.train_tinyllama_vision2text \
  --files "dataset/raw_data/nc_*/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --mae_ckpt checkpoints/satswinmae_epoch59.pt \
  --caption_csv dataset/weather/weather_jan_may_2024.csv \
  --use_qlora --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
  --batch_size 16 --epochs 50 --lr 0.0001
```

### 15.2 TinyLlama Inference

Generate captions for a specific date:
```bash
python infer_tinyllama_vision2text.py \
  --files "dataset/raw_data/nc_*/*.nc" \
  --variables u10 v10 r sp ssrd t cp \
  --date 2024-03-15 \
  --mae_ckpt checkpoints/satswinmae_epoch59.pt \
  --adapter_ckpt checkpoints_v2t/tinyllama_adapter_final.pt \
  --max_new_tokens 128 --temperature 0.7
```

### 15.3 HRM Vision-to-Text Training

```bash
python -m vision_text.train_hrm_vision2text \
  --files "dataset/raw_data/nc_*/*.nc" \
  --variables t2m sp \
  --mae_ckpt checkpoints/satswinmae_epoch10.pt \
  --caption_csv dataset/weather/weather_august_2024.csv \
  --epochs 10 --batch_size 4 --lr 1e-4
```

---

*End of technical overview.*
