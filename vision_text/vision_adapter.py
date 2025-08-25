
import torch
import torch.nn as nn

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(d_model*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model*mlp_ratio), d_model),
        )
    def forward(self, q, kv):
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + out
        q = q + self.ff(q)
        return q

class PerceiverResampler(nn.Module):
    """Learnable queries attend over visual token sequence to produce M prompts."""
    def __init__(self, d_model, n_latents=32, n_layers=2, n_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
    def forward(self, kv):
        B, L, D = kv.shape
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B,M,D)
        for blk in self.blocks:
            lat = blk(lat, kv)
        return lat  # (B,M,D)

class VisionPrefixer(nn.Module):
    """SatSwinMAE -> (B,L,D_vis) -> project -> resample to (B,M,d_model)."""
    def __init__(self, mae_encoder, d_vis, d_model, n_latents=32, n_layers=2, n_heads=8, dropout=0.0):
        super().__init__()
        self.mae = mae_encoder
        self.proj = nn.Linear(d_vis, d_model)
        self.resampler = PerceiverResampler(d_model, n_latents=n_latents, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

    @torch.no_grad()
    def encode_vis_tokens(self, cube):
        # cube: (B,C,T,H,W)
        # Expect SatSwinMAE to expose encode_tokens() -> (B,L,D_vis)
        z, _ = self.mae.encode_tokens(cube)
        return z

    def forward(self, cube):
        with torch.no_grad():
            z = self.encode_vis_tokens(cube)  # (B,L,D_vis)
        z = self.proj(z)                      # (B,L,d_model)
        prompts = self.resampler(z)           # (B,M,d_model)
        return prompts

