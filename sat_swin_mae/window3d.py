import torch
import torch.nn.functional as F

def window_partition(x, window_size):
    # x: (B, T, H, W, C) -> (B*nW, Wt*Wh*Ww, C)
    B, T, H, W, C = x.shape
    Wt, Wh, Ww = window_size
    pad_t = (Wt - T % Wt) % Wt
    pad_h = (Wh - H % Wh) % Wh
    pad_w = (Ww - W % Ww) % Ww
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0,0, 0,pad_w, 0,pad_h, 0,pad_t))
        T += pad_t; H += pad_h; W += pad_w
    x = x.view(B, T//Wt, Wt, H//Wh, Wh, W//Ww, Ww, C)
    x = x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, Wt*Wh*Ww, C)
    return x, (pad_t, pad_h, pad_w), (T, H, W)

def window_reverse(windows, window_size, T, H, W, B):
    Wt, Wh, Ww = window_size
    C = windows.shape[-1]
    windows = windows.view(B, T//Wt, H//Wh, W//Ww, Wt, Wh, Ww, C)
    x = windows.permute(0,1,4,2,5,3,6,7).contiguous().view(B, T, H, W, C)
    return x

def compute_attn_mask(T, H, W, window_size, shift_size, device):
    Wt, Wh, Ww = window_size
    St, Sh, Sw = shift_size
    if St==0 and Sh==0 and Sw==0:
        return None
    img_mask = torch.zeros((1, T, H, W, 1), device=device)
    cnt = 0
    t_slices = (slice(0, -Wt), slice(-Wt, -St), slice(-St, None)) if St>0 else (slice(0,T),)
    h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None)) if Sh>0 else (slice(0,H),)
    w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None)) if Sw>0 else (slice(0,W),)
    for t in t_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, t, h, w, :] = cnt
                cnt += 1
    mask_windows, _, _ = window_partition(img_mask, window_size)  # (nW, Ws^3, 1)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
