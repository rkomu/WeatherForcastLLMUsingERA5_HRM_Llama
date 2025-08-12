import math
import torch
import torch.nn as nn

def get_3d_sincos_pos_embed(C, T, H, W, device=None):
    """3D sin-cos positional embeddings -> (T*H*W, C)."""
    def get_1d_pos_embed(c, length):
        pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, c, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / c))
        pe = torch.zeros(length, c, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    c_t = C // 3
    c_h = C // 3
    c_w = C - c_t - c_h
    pt = get_1d_pos_embed(c_t, T)
    ph = get_1d_pos_embed(c_h, H)
    pw = get_1d_pos_embed(c_w, W)

    pe = torch.zeros(T, H, W, C, device=device)
    pe[..., :c_t] = pt[:, None, None, :]
    pe[..., c_t:c_t+c_h] = ph[None, :, None, :]
    pe[..., c_t+c_h:] = pw[None, None, :, :]
    return pe.view(T*H*W, C)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
