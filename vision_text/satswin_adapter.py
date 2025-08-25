
"""
vision_text/satswin_adapter.py

Normal-import SatSwin feature adapter for HRM Vision2Text.
No CLI flag is required. Training/inference scripts can:

    from vision_text.satswin_adapter import build_encoder
    encoder, feat_dim = build_encoder(device=device, **kwargs)

It wraps sat_swin_mae.model.SatSwinMAE and exposes a forward that takes
ERA5 cubes: Tensor (B, C, T, H, W) and returns pooled features (B, D).
"""

from typing import Tuple, Optional, Dict, Any
import os
import torch
import torch.nn as nn

try:
    from sat_swin_mae.model import SatSwinMAE
except Exception as e:
    raise RuntimeError("Could not import SatSwinMAE from sat_swin_mae.model. "
                       "Ensure the package folder 'sat_swin_mae' exists and is on PYTHONPATH.") from e


class SatSwinFeatureAdapter(nn.Module):
    """
    Wraps SatSwinMAE encoder to produce a single feature vector per cube.
    - Expects input cubes with shape (B, C, T, H, W) in float32.
    - Uses `encode_tokens` to get top-stage tokens, then pools ("mean" or "cls").
    - Frozen by default (set freeze=False to finetune encoder).
    """
    def __init__(
        self,
        in_chans: int,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2),
        num_heads: Tuple[int, ...] = (3, 6),
        window_size: Tuple[int, int, int] = (2, 8, 8),
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        mask_ratio: float = 0.75,
        pool: str = "mean",
        ckpt: Optional[str] = None,
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.pool = str(pool).lower()
        self.model = SatSwinMAE(
            in_chans=in_chans,
            out_chans=in_chans,  # decoder unused for features
            embed_dim=embed_dim,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            window_size=tuple(window_size),
            patch_size=tuple(patch_size),
            mask_ratio=mask_ratio
        )
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[SatSwinFeatureAdapter] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
        else:
            if ckpt:
                print(f"[SatSwinFeatureAdapter] WARNING: ckpt not found at {ckpt}; using randomly initialized encoder.")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        if device is not None:
            self.to(device)

        # Feature dimension = top-stage encoder dim = embed_dim*(2**(len(depths)-1))
        self.feat_dim = int(embed_dim * (2 ** (len(depths) - 1)))

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns tokens from top encoder stage: (B, L, C_enc)
        """
        tokens, _ = self.model.encoder(x, mask=None), None  # encoder returns list of stage features
        top = tokens[-1]  # (B, T', H', W', C_enc)
        B, T, H, W, C = top.shape
        return top.view(B, T * H * W, C)

    def _pool_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        # tok: (B, L, C)
        if self.pool == "cls" and tok.size(1) > 0:
            return tok[:, 0, :]
        return tok.mean(dim=1)

    def forward(self, cubes: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        cubes: (B, C, T, H, W)
        valid_mask: optional (B, T, H, W) -> if provided, masked tokens are excluded from mean pooling.
        """
        if not torch.is_tensor(cubes):
            raise TypeError(f"Expected cubes as Tensor, got {type(cubes)}")
        if cubes.dim() != 5:
            raise ValueError(f"Expected cubes shape (B,C,T,H,W), got {tuple(cubes.shape)}")

        # Use SatSwinMAE's encode_tokens (no masking at inference)
        x_top = self.model.encoder(cubes, mask=None)[-1]  # (B, T', H', W', C_enc)
        B, T, H, W, C = x_top.shape
        tok = x_top.view(B, T * H * W, C)  # (B, L, C)

        if valid_mask is not None:
            # Downsample valid mask to token grid via nearest (simple & fast)
            vm = valid_mask.float().unsqueeze(1)  # (B,1,T,H,W)
            vm_ds = torch.nn.functional.interpolate(
                vm, size=(T, H, W), mode="nearest"
            ).view(B, T * H * W)  # (B, L)
            vm_sum = vm_ds.sum(dim=1, keepdim=True).clamp_min(1.0)  # avoid div-by-zero
            feat = (tok * vm_ds.unsqueeze(-1)).sum(dim=1) / vm_sum
        else:
            feat = tok.mean(dim=1)  # (B, C_enc)

        return feat.to(torch.float32)


def build_encoder(device: torch.device, **kwargs) -> Tuple[nn.Module, int]:
    """
    Build the SatSwinFeatureAdapter with kwargs, return (module, feat_dim).
    Supported kwargs:
      - in_chans (int, required)
      - embed_dim (int, default 96)
      - depths (list/tuple of ints, default [2,2])
      - num_heads (list/tuple of ints, default [3,6])
      - window_size (list/tuple, default [2,8,8])
      - patch_size (list/tuple, default [2,4,4])
      - mask_ratio (float, default 0.75)
      - pool ("mean"|"cls", default "mean")
      - ckpt (str, optional)
      - freeze (bool, default True)
    """
    def _get_seq(name, default):
        v = kwargs.get(name, default)
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return tuple(int(x) for x in (v, v)) if isinstance(v, int) else default

    in_chans   = int(kwargs.get("in_chans", 1))
    embed_dim  = int(kwargs.get("embed_dim", 96))
    depths     = _get_seq("depths", (2, 2))
    num_heads  = _get_seq("num_heads", (3, 6))
    window_sz  = _get_seq("window_size", (2, 8, 8))
    patch_sz   = _get_seq("patch_size", (2, 4, 4))
    mask_ratio = float(kwargs.get("mask_ratio", 0.75))
    pool       = str(kwargs.get("pool", "mean"))
    ckpt       = kwargs.get("ckpt", None)
    freeze     = bool(kwargs.get("freeze", True))

    enc = SatSwinFeatureAdapter(
        in_chans=in_chans, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
        window_size=window_sz, patch_size=patch_sz, mask_ratio=mask_ratio,
        pool=pool, ckpt=ckpt, freeze=freeze, device=device
    )
    return enc, enc.feat_dim
