import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


def robust_normalize_C(C, eps=1e-6, p=0.98):
    """
    C: (n_ref, B, H, W)
    """
    # Compute percentile per (ref, batch)
    flat = C.flatten(-2)
    scale_p = torch.quantile(flat, p, dim=-1, keepdim=True)
    scale_p = scale_p.unsqueeze(-1)

    # Fallback to mean if percentile is too small
    scale_m = C.mean(dim=(-2, -1), keepdim=True)

    scale = torch.where(
        scale_p > eps,
        scale_p,
        scale_m
    )

    return C / (scale + eps), scale

class RefAdaptivePercentile(nn.Module):
    def __init__(self, n_ref, p_init=0.95):
        super().__init__()
        self.p_raw = nn.Parameter(
            torch.full((n_ref,), p_init)
        )

    def forward(self):
        return torch.clamp(self.p_raw, 0.7, 0.99)


def adaptive_normalize_C_ref(C, p_module, eps=1e-6):
    p = p_module()                            # (n_ref,)
    flat = C.flatten(-2)
    scales = []
    for r in range(C.shape[0]):
        s = torch.quantile(flat[r], p[r], dim=-1, keepdim=True)
        scales.append(s.unsqueeze(-1))
    scale = torch.stack(scales, dim=0)
    scale = torch.clamp(scale, min=eps)

    return C / scale, scale

def sliding_window_ref_aware(
    C,                     # (n_ref, B, H, W)
    denoiser,
    p_module,              # RefAdaptivePercentile
    win_size=5,
    eps=1e-6
):
    """
    Reference-aware sliding-window 3D CNN denoiser
    with adaptive percentile normalization.

    Args:
        C:         (n_ref, B, H, W)
        denoiser:  RefAware3DCNN
        p_module:  RefAdaptivePercentile
        win_size:  sliding window size along B
    Returns:
        C_denoised: (n_ref, B, H, W)
    """
    n_ref, B, H, W = C.shape
    pad = win_size // 2

    # ------------------------------------------------
    # 1. Adaptive percentile normalization (NO inplace)
    # ------------------------------------------------
    Cn, scale = adaptive_normalize_C_ref(C, p_module, eps=eps)
    # Cn, scale: both differentiable

    # ------------------------------------------------
    # 2. Pad along B dimension
    # ------------------------------------------------
    C_pad = F.pad(
        Cn,
        (0, 0, 0, 0, pad, pad),
        mode="replicate"
    )  # (n_ref, B+2p, H, W)

    # ------------------------------------------------
    # 3. Sliding-window denoising
    # ------------------------------------------------
    out_slices = []

    for b in range(B):
        win = C_pad[:, b:b + win_size]      # (n_ref, win, H, W)
        win = win.unsqueeze(0)              # (1, n_ref, win, H, W)

        out = denoiser(win).squeeze(0)      # (n_ref, win, H, W)
        out_slices.append(out[:, pad])      # center slice

    C_out = torch.stack(out_slices, dim=1)  # (n_ref, B, H, W)

    # ------------------------------------------------
    # 4. De-normalize (NO inplace)
    # ------------------------------------------------
    return C_out * scale


def checkpointed_denoiser(
    C,
    denoiser,
    p_module,
    win_size
):
    """
    Gradient-checkpointed reference-aware sliding-window denoiser
    with adaptive percentile normalization.
    """
    def fn(x):
        return sliding_window_ref_aware(
            x,
            denoiser=denoiser,
            p_module=p_module,
            win_size=win_size
        )

    return checkpoint.checkpoint(fn, C)
