import torch
import torch.nn as nn
import torch.nn.functional as F
from .patch_embed import PatchEmbed3D, PatchMerging3D, PatchExpanding3D
from .layers import SwinBlock3D
from .masking import random_masking
from .utils import get_3d_sincos_pos_embed

class Encoder3D(nn.Module):
    def __init__(self, in_chans, embed_dim=96, depths=(2,2), num_heads=(3,6), window_size=(2,7,7), patch_size=(2,4,4), drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_chans, embed_dim, patch_size)
        self.pos_embed = None
        self.stages = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        dim = embed_dim
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        idx = 0
        for i, d in enumerate(depths):
            blocks = []
            for j in range(d):
                shift = (0,0,0) if (j % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
                blocks.append(SwinBlock3D(dim, input_resolution=None, num_heads=num_heads[i], window_size=window_size, shift_size=shift, drop_path=dp_rates[idx]))
                idx += 1
            self.stages.append(nn.Sequential(*blocks))
            if i < len(depths)-1:
                self.merge_layers.append(PatchMerging3D(dim, dim*2))
                dim = dim*2
        self.out_dims = [embed_dim] + [embed_dim*(2**i) for i in range(1, len(depths))]

    def forward(self, x, mask=None):
        x, (T, H, W) = self.patch_embed(x)  # (B, T', H', W', C)
        B, T, H, W, C = x.shape
        if self.pos_embed is None or self.pos_embed.shape[-1] != C or self.pos_embed.shape[0] != T*H*W:
            self.pos_embed = get_3d_sincos_pos_embed(C, T, H, W, x.device)
        x = x + self.pos_embed.view(1, T, H, W, C)
        if mask is not None:
            if not hasattr(self, "mask_token"):
                self.mask_token = nn.Parameter(torch.zeros(1,1,1,1,C))
                nn.init.normal_(self.mask_token, std=0.02)
            # Do not reassign self.mask_token; use a local variable for device transfer
        mask_token = self.mask_token
        if mask_token.device != x.device:
            mask_token = mask_token.to(x.device)
        x = torch.where(mask.unsqueeze(-1), mask_token.expand_as(x), x)
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.merge_layers):
                x = self.merge_layers[i](x)
        return feats  # low->high

class Decoder3D(nn.Module):
    def __init__(self, out_chans, embed_dims, depths=(2,2), num_heads=(3,6), window_size=(2,7,7), drop_path_rate=0.1):
        super().__init__()
        self.stages = nn.ModuleList()
        self.expand_layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        dims_up = list(reversed(embed_dims))
        depths_up = list(reversed(depths))
        num_heads_up = list(reversed(num_heads))
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_up))]
        idx = 0
        for i in range(len(dims_up)-1):
            blocks = []
            for j in range(depths_up[i]):
                shift = (0,0,0) if (j % 2 == 0) else (window_size[0]//2, window_size[1]//2, window_size[2]//2)
                blocks.append(SwinBlock3D(dims_up[i], input_resolution=None, num_heads=num_heads_up[i], window_size=window_size, shift_size=shift, drop_path=dp_rates[idx]))
                idx += 1
            self.stages.append(nn.Sequential(*blocks))
            self.expand_layers.append(PatchExpanding3D(dims_up[i], dims_up[i+1]))
            self.skip_projs.append(nn.Linear(dims_up[i+1]*2, dims_up[i+1]))
        self.head = nn.Sequential(nn.LayerNorm(dims_up[-1]), nn.Linear(dims_up[-1], out_chans))

    def forward(self, feats):
        feats_up = list(reversed(feats))
        x = feats_up[0]
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            x = self.expand_layers[i](x)
            skip = feats_up[i+1]
            B,T,H,W,C = x.shape
            x = torch.cat([x, skip], dim=-1).view(B*T*H*W, -1)
            x = self.skip_projs[i](x).view(B, T, H, W, -1)
        B,T,H,W,C = x.shape
        x = self.head(x.view(B*T*H*W, C)).view(B,T,H,W,-1)
        return x

class SatSwinMAE(nn.Module):
    def __init__(self, in_chans, out_chans, embed_dim=96, depths=(2,2), num_heads=(3,6),
                 window_size=(2,7,7), patch_size=(2,4,4), mask_ratio=0.75, drop_path_rate=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = Encoder3D(in_chans, embed_dim, depths, num_heads, window_size, patch_size, drop_path_rate)
        embed_dims = [embed_dim*(2**i) for i in range(len(depths))]
        self.decoder = Decoder3D(out_chans, embed_dims, depths, num_heads, window_size, drop_path_rate)
        self.patch_size = patch_size

    def forward(self, x, compute_loss=True, valid_mask=None):
        B, C, T, H, W = x.shape
        pt, ph, pw = self.patch_size

        mask = random_masking(B, T//pt, H//ph, W//pw, self.mask_ratio, device=x.device)
        feats = self.encoder(x, mask=mask)
        recon = self.decoder(feats)  # (B, T', H', W', C_out)

        recon = F.interpolate(
            recon.permute(0,4,1,2,3),
            scale_factor=(pt, ph, pw),
            mode='trilinear',
            align_corners=False
        ).permute(0,2,3,4,1)  # (B, T, H, W, C_out)

        if not compute_loss:
            return recon.permute(0,4,1,2,3)

        x_tgt = x.permute(0,2,3,4,1)  # (B, T, H, W, C)
        loss_map = (recon - x_tgt) ** 2  # (B, T, H, W, C)

        # Build full-res mask from patch mask
        mask_full = mask.unsqueeze(-1)                         # (B, T', H', W', 1)
        mask_full = mask_full.repeat_interleave(pt, 1)\
                               .repeat_interleave(ph, 2)\
                               .repeat_interleave(pw, 3)       # (B, T, H, W, 1)
        mask_M = mask_full.squeeze(-1)                         # masked locations
        mask_U = ~mask_M                                       # unmasked locations

        # Validity (exclude NaN/land wave gaps, etc.)
        if valid_mask is not None:                             # valid_mask: (B,1,T,H,W) or (B,T,H,W)
            if valid_mask.dim() == 5:
                v = valid_mask.squeeze(1)                      # (B,T,H,W)
            else:
                v = valid_mask
            mask_M = mask_M & v
            mask_U = mask_U & v

        # Compute safe means (avoid empty selections -> NaN)
        loss = 0.0
        if mask_M.any():
            loss += 0.9 * loss_map[mask_M].mean()
        if mask_U.any():
            loss += 0.1 * loss_map[mask_U].mean()
        if not mask_M.any() and not mask_U.any():
            # fallback: everything invalid? just average over finite values
            finite = torch.isfinite(loss_map)
            loss = loss_map[finite].mean()

        return loss, recon.permute(0,4,1,2,3)

    @torch.no_grad()
    def encode_tokens(self, x):
        """Return the top-stage encoder feature map as tokens (B, L, C_enc)."""
        feats = self.encoder(x, mask=None)
        x_top = feats[-1]               # (B, T',H',W', C)
        B,T,H,W,C = x_top.shape
        return x_top.view(B, T*H*W, C), (T,H,W)
