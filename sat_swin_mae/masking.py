import torch

def random_masking(B, T, H, W, mask_ratio, device=None):
    num = T*H*W
    num_mask = int(num * mask_ratio)
    num_mask = max(1, min(num-1, num_mask))  # keep at least 1 unmasked and 1 masked
    mask = torch.zeros(B, num, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(num, device=device)[:num_mask]
        mask[b, idx] = True
    return mask.view(B, T, H, W)
