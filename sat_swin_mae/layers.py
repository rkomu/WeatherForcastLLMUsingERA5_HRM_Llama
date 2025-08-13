import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DropPath
from .window3d import window_partition, window_reverse, compute_attn_mask

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim; self.window_size = window_size; self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        Wt, Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*Wt-1)*(2*Wh-1)*(2*Ww-1), num_heads))
        coords_t = torch.arange(Wt); coords_h = torch.arange(Wh); coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1,2,0).contiguous()
        rel[:, :, 0] += Wt - 1; rel[:, :, 1] += Wh - 1; rel[:, :, 2] += Ww - 1
        rel[:, :, 0] *= (2*Wh-1)*(2*Ww-1)
        rel[:, :, 1] *= (2*Ww-1)
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, attn_mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        rel_bias = rel_bias.permute(2,0,1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x

class SwinBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(2,7,7),
                 shift_size=(0,0,0), mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), drop=drop)
        self.attn_mask = None

    def forward(self, x):
        B, T, H, W, C = x.shape
        Wt, Wh, Ww = self.window_size
        St, Sh, Sw = self.shift_size

        shortcut = x
        x = self.norm1(x.view(B*T*H*W, C)).view(B, T, H, W, C)
        if St or Sh or Sw:
            x = torch.roll(x, shifts=(-St, -Sh, -Sw), dims=(1,2,3))
        x_windows, _, (Tpad, Hpad, Wpad) = window_partition(x, self.window_size)
        if self.attn_mask is None or self.attn_mask.shape[1] != x_windows.shape[1]:
            self.attn_mask = compute_attn_mask(Tpad, Hpad, Wpad, self.window_size, self.shift_size, x.device)
        attn_windows = self.attn(x_windows, attn_mask=self.attn_mask)
        x = window_reverse(attn_windows, self.window_size, Tpad, Hpad, Wpad, B)
        if St or Sh or Sw:
            x = torch.roll(x, shifts=(St, Sh, Sw), dims=(1,2,3))
        x = x[:, :T, :H, :W, :]
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x.view(B*T*H*W, C)).view(B, T, H, W, C)))
        return x

