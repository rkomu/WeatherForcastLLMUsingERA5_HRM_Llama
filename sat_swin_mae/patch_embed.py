import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, patch_size=(2,4,4)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.permute(0,2,3,4,1).contiguous()
        x = self.norm(x.view(B*T*H*W, C)).view(B, T, H, W, C)
        return x, (T, H, W)

class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim*2
        self.reduction = nn.Linear(dim*8, out_dim, bias=False)
        self.norm = nn.LayerNorm(dim*8)

    def forward(self, x):
        B, T, H, W, C = x.shape
        if T%2 or H%2 or W%2:
            x = F.pad(x, (0,0, 0,W%2, 0,H%2, 0,T%2))
            B, T, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 0::2, 0::2, 1::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0,x1,x2,x3,x4,x5,x6,x7], dim=-1)
        x = self.norm(x.view(B*(T//2)*(H//2)*(W//2), -1))
        x = self.reduction(x).view(B, T//2, H//2, W//2, -1)
        return x

class PatchExpanding3D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        B,T,H,W,C = x.shape
        x = x.permute(0,4,1,2,3).contiguous()
        x = self.proj(x)
        B,C2,T2,H2,W2 = x.shape
        x = x.permute(0,2,3,4,1).contiguous()
        x = self.norm(x.view(B*T2*H2*W2, C2)).view(B,T2,H2,W2,C2)
        return x
