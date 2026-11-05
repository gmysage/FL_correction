import torch
import math
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn as nn


def _rotated_hw(H, W, theta):
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    H_new = int(H * cos_t + W * sin_t + 0.5)
    W_new = int(W * cos_t + H * sin_t + 0.5)
    return H_new, W_new

def torch_rot3D(img_raw, rot_angle, order=1):
    """
    Rotate 2D or 3D or 4D (set of 3D) image with angle = rot_angle.
    Rotates counter-clockwise. Safe for PyTorch autograd.

    Parameters:
    -----------
    img_raw:    2D, 3D, or 4D torch.Tensor
    rot_angle:  float (in degrees)
    order:      int (1 for bilinear interpolation, 0 for nearest neighbor)

    Returns:
    --------
    torch.Tensor with the same shape as input image, all values >= 0
    """
    # Clone to avoid mutating original, ensure float32
    img = img_raw.clone().to(dtype=torch.float32)

    # Safe out-of-place replacement for rm_nan
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if rot_angle == 0:
        return img

    s = img.shape
    ndim = len(s)

    if ndim > 4:
        raise ValueError('Error! Input image has dimension > 4')

    # Map interpolation order to torchvision Interp Modes
    # order=0 is nearest neighbor, order=1 (or others) is bilinear [1]
    interpolation = TF.InterpolationMode.NEAREST if (rot_angle % 90 == 0 or order == 0) else TF.InterpolationMode.BILINEAR

    # --- 2D Case: Shape (H, W) ---
    if ndim == 2:
        # F.rotate expects [C, H, W] or [B, C, H, W], so we unsqueeze to add a channel dim [1]
        img_rot = TF.rotate(img.unsqueeze(0), rot_angle, interpolation=interpolation).squeeze(0)

    # --- 3D Case: Shape (D, H, W) ---
    # Rotating along axes=[1, 2] means D behaves exactly like a Batch dimension [1]
    elif ndim == 3:
        # F.rotate handles a batch/stack of images out-of-the-box if we add a channel dimension [1]
        # (D, H, W) -> unsqueeze channel -> (D, 1, H, W) -> rotate -> squeeze channel -> (D, H, W)
        img_rot = TF.rotate(img.unsqueeze(1), rot_angle, interpolation=interpolation).squeeze(1)

    # --- 4D Case: Shape (N, D, H, W) ---
    # Rotating each 3D set along axes=[1, 2] means N * D can be flattened into a single batch dimension
    elif ndim == 4:
        # Flatten N and D into one batch dimension to process everything in parallel without loops
        flattened_img = img.view(-1, 1, s[2], s[3])  # Shape: (N*D, 1, H, W)
        rotated_flat = TF.rotate(flattened_img, rot_angle, interpolation=interpolation)
        img_rot = rotated_flat.view(s)  # Unflatten back to original shape: (N, D, H, W)

    # ✅ FIX: Avoid in-place tracking error by using an out-of-place clamp [2]
    img_rot = torch.clamp(img_rot, min=0.0)

    return img_rot

def torch_rotate_image_nd(
        x,
        angle,
        enable_crop=True,
        align_corners=False,
        mode="bilinear",
        padding_mode="zeros",
):
    """
    ### It is equivalent to torch_rot3D ###

    Rotate an N-D image tensor using PyTorch.

    Args
    ----
    x : torch.Tensor
        Shape (..., H, W)
    angle : float
        Rotation angle in degrees (positive = CCW)
    enable_crop : bool
        True  -> output size == input size (cropped)
        False -> expand canvas to avoid cropping
    align_corners : bool
        Passed to affine_grid / grid_sample

    Returns
    -------
    torch.Tensor
        Rotated tensor
    """

    orig_shape = x.shape
    device = x.device
    dtype = x.dtype
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)

    if angle == 0:
        return x
    # ----------------------------
    # Normalize shape to (N, C, H, W)
    # ----------------------------
    if x.dim() == 2:  # (H, W)
        x = x[None, None]
        batch_shape = ()
        #C = 1
    elif x.dim() == 3:  # (C, H, W)
        x = x[None]
        batch_shape = ()
        #C = x.shape[1]
    elif x.dim() >= 4:  # (..., H, W)
        batch_shape = x.shape[:-2]
        #C = 1
        x = x.reshape(-1, 1, *x.shape[-2:])
    else:
        raise ValueError("Input must have at least 2 dimensions")

    N, C, H, W = x.shape
    if angle % 90 == 0:
        mode = "nearest"
    theta = math.radians(angle)
    # ----------------------------
    # Output canvas size
    # ----------------------------
    if enable_crop:
        H_out, W_out = H, W
    else:
        H_out, W_out = _rotated_hw(H, W, theta)

    # ----------------------------
    # Rotation matrix
    # ----------------------------
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    theta_mat = torch.tensor(
        [[cos_t, -sin_t, 0.0],
         [sin_t, cos_t, 0.0]],
        dtype=dtype,
        device=device,
    )

    theta_mat = theta_mat.unsqueeze(0).repeat(N, 1, 1)

    # ----------------------------
    # Grid + sampling
    # ----------------------------
    grid = F.affine_grid(
        theta_mat,
        size=(N, C, H_out, W_out),
        align_corners=align_corners,
    )

    y = F.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # ----------------------------
    # Restore original shape
    # ----------------------------
    if len(batch_shape) > 0:
        y = y.reshape(*batch_shape, H_out, W_out)
    else:
        y = y.reshape(orig_shape)

    return y


def torch_rot90_3D(img_raw, ax=0, mode='c-clock'):
    '''
    ax: rotation axes
    ax=0: positive direction from bottom --> top
    ax=1: positive direction from front --> back (this is un-conventional to righ-hand-rule)
    ax=2: positive direction from left --> right

    mode:
        'clock': rotate clockwise --> "-90 degree"
        'c-clock': rotate count-clockwise --> "90 degree"
    '''
    img_r = img_raw.clone()
    if ax == 0:
        img_r = torch.transpose(img_r, 1, 2)
        if mode == 'clock':
            img_r = img_r.flip(dims=(2,))
        elif mode == 'c-clock':
            img_r = img_r.flip(dims=(1,))
    if ax == 1:
        img_r = torch.transpose(img_r, 0, 2)
        if mode == 'clock':
            img_r = img_r.flip(dims=(2,))
        elif mode == 'c-clock':
            img_r = img_r.flip(dims=(0,))
    if ax == 2:
        img_r = torch.transpose(img_r, 0, 1)
        if mode == 'clock':
            img_r = img_r.flip(dims=(0,))
        elif mode == 'c-clock':
            img_r = img_r.flip(dims=(1,))
    return img_r


def torch_rot90_4D(img4D, ax=0, mode='c-clock'):
    s = img4D.shape
    img4D_r = torch.zeros_like(img4D)
    for i in range(s[0]):
        img4D_r[i] = torch_rot90_3D(img4D[i], ax, mode)
    return img4D_r


def torch_rot90_4D_tensor(img4D, ax=0, mode='c-clock'):
    """
    Vectorized, loop-free 4D volumetric rotation.
    Matches your exact 3D custom axes logic but runs 100% out-of-place
    and is safe for training/gradients.

    img4D shape: (n_elem, n_sli, H, W)
    """
    # Create a reference pointer (out-of-place assignment)
    img_r = img4D

    if ax == 0:
        # 3D: transpose(1, 2) -> 4D: transpose(2, 3)
        img_r = torch.transpose(img_r, 2, 3)
        if mode == 'clock':
            img_r = img_r.flip(dims=(3,))  # 3D dim 2 -> 4D dim 3
        elif mode == 'c-clock':
            img_r = img_r.flip(dims=(2,))  # 3D dim 1 -> 4D dim 2

    elif ax == 1:
        # 3D: transpose(0, 2) -> 4D: transpose(1, 3)
        img_r = torch.transpose(img_r, 1, 3)
        if mode == 'clock':
            img_r = img_r.flip(dims=(3,))  # 3D dim 2 -> 4D dim 3
        elif mode == 'c-clock':
            img_r = img_r.flip(dims=(1,))  # 3D dim 0 -> 4D dim 1

    elif ax == 2:
        # 3D: transpose(0, 1) -> 4D: transpose(1, 2)
        img_r = torch.transpose(img_r, 1, 2)
        if mode == 'clock':
            img_r = img_r.flip(dims=(1,))  # 3D dim 0 -> 4D dim 1
        elif mode == 'c-clock':
            img_r = img_r.flip(dims=(2,))  # 3D dim 1 -> 4D dim 2

    return img_r


class DifferentiableNLM2D(nn.Module):
    """
    Differentiable non-local-means-like denoiser for 2D tensors.

    Input:
        x: (B, C, H, W)

    Output:
        y: (B, C, H, W)

    Notes:
        - Uses patch similarity inside a search window.
        - Autograd is preserved because it is built from PyTorch ops only.
        - Computational cost grows with search_size^2.

    example usage:
        nlm = DifferentiableNLM2D(patch_size=3, search_size=7, h=0.08).cuda()
        x = torch.rand(2, 1, 128, 128, device="cuda", requires_grad=True)
        y = nlm(x)
        y = y.clamp(y, min=0.0)
    """
    def __init__(
        self,
        patch_size: int = 3,
        search_size: int = 7,
        h: float = 0.1,
        sigma_spatial: float = 3.0,
        learnable_h: bool = False,
        pad_mode: str = "reflect",
        eps: float = 1e-8,
    ):
        super().__init__()
        assert patch_size % 2 == 1, "patch_size must be odd"
        assert search_size % 2 == 1, "search_size must be odd"

        self.patch_size = patch_size
        self.search_size = search_size
        self.sigma_spatial = float(sigma_spatial)
        self.pad_mode = pad_mode
        self.eps = eps

        h0 = torch.tensor(float(h)).log()
        if learnable_h:
            self.log_h = nn.Parameter(h0)
        else:
            self.register_buffer("log_h", h0)

    def _h2(self):
        h = self.log_h.exp()
        return (h * h).clamp_min(self.eps)

    def forward(self, x: torch.Tensor, guide: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, C, H, W), got {tuple(x.shape)}")

        if guide is None:
            guide = x
        if guide.shape != x.shape:
            raise ValueError(f"guide must have the same shape as x. Got {tuple(guide.shape)} vs {tuple(x.shape)}")

        B, C, H, W = x.shape
        p = self.patch_size // 2
        s = self.search_size // 2
        total_pad = p + s

        # Pad once so every candidate patch is valid.
        x_pad = F.pad(x, (total_pad, total_pad, total_pad, total_pad), mode=self.pad_mode)
        g_pad = F.pad(guide, (total_pad, total_pad, total_pad, total_pad), mode=self.pad_mode)

        # Reference patches from the center crop.
        ref_crop = g_pad[:, :, s:s + H + 2 * p, s:s + W + 2 * p]
        ref_patches = F.unfold(ref_crop, kernel_size=self.patch_size)  # (B, C*P*P, H*W)

        candidate_values = []
        candidate_logits = []

        h2 = self._h2()
        spatial_denom = 2.0 * (self.sigma_spatial ** 2) + self.eps

        for dy in range(-s, s + 1):
            for dx in range(-s, s + 1):
                oy = dy + s
                ox = dx + s

                # Candidate crop aligned with the output grid.
                cand_val_crop = x_pad[:, :, oy:oy + H + 2 * p, ox:ox + W + 2 * p]
                cand_guide_crop = g_pad[:, :, oy:oy + H + 2 * p, ox:ox + W + 2 * p]

                cand_patches = F.unfold(cand_guide_crop, kernel_size=self.patch_size)  # (B, C*P*P, H*W)

                # Patch distance per output pixel.
                dist = (ref_patches - cand_patches).pow(2).mean(dim=1)  # (B, H*W)

                # Spatial prior on the offset.
                spatial_bias = -float(dy * dy + dx * dx) / spatial_denom

                # Larger logit = more weight.
                logits = -(dist / h2) + spatial_bias
                candidate_logits.append(logits)

                # Center pixels from the candidate crop.
                candidate_values.append(cand_val_crop[:, :, p:p + H, p:p + W])  # (B, C, H, W)

        # Stack over candidate offsets.
        logits = torch.stack(candidate_logits, dim=1)  # (B, K, H*W)
        weights = torch.softmax(logits, dim=1)         # (B, K, H*W)

        values = torch.stack(candidate_values, dim=1)  # (B, K, C, H, W)

        weights = weights.view(B, -1, 1, H, W)          # (B, K, 1, H, W)
        out = (weights * values).sum(dim=1)             # (B, C, H, W)

        return out

def torch_nlm_single_image(xx, nlm):
    """
    xx:
        (H,1,W)  or
        (1,H,W)
    """

    if xx.shape[0] == 1:
        # (1,H,W)
        x = xx.unsqueeze(0)
        y = nlm(x)
        y = torch.clamp(y, min=0)
        return y.squeeze(0)

    elif xx.shape[1] == 1:
        # (H,1,W)
        x = xx.permute(1,0,2).unsqueeze(0)
        y = nlm(x)
        y = torch.clamp(y, min=0)
        return y.squeeze(0).permute(1,0,2)

    else:
        raise ValueError(f"Unexpected shape {xx.shape}")

class DifferentiableNLM3D(nn.Module):
    """
    Differentiable non-local-means-like denoiser for 3D volumes.

    Input:
        x:     (B, C, D, H, W)
        guide: (B, C, D, H, W) or None
               If None, self-denoising is performed (guide = x)

    Output:
        y:     (B, C, D, H, W)

    Notes:
        - Uses cubic patches and a cubic search window.
        - Autograd is preserved.
        - Cost grows quickly with search_size^3.

    example:
        nlm3d = DifferentiableNLM3D(
                    patch_size=3,
                    search_size=5,
                    h=0.06,
                    sigma_spatial=2.0,
                    learnable_h=False,
                    ).to(device)

        # x: (B, C, D, H, W)
        y = nlm3d(x)
        or:
        x = volume.unsqueeze(0).unsqueeze(0)   # (1, 1, D, H, W)
        y = nlm3d(x)
        y = y.squeeze(0).squeeze(0)
    """
    def __init__(
        self,
        patch_size: int = 3,
        search_size: int = 5,
        h: float = 0.1,
        sigma_spatial: float = 3.0,
        learnable_h: bool = False,
        pad_mode: str = "replicate",
        eps: float = 1e-8,
    ):
        super().__init__()
        assert patch_size % 2 == 1, "patch_size must be odd"
        assert search_size % 2 == 1, "search_size must be odd"

        self.patch_size = patch_size
        self.search_size = search_size
        self.sigma_spatial = float(sigma_spatial)
        self.pad_mode = pad_mode
        self.eps = eps

        h0 = torch.tensor(float(h)).log()
        if learnable_h:
            self.log_h = nn.Parameter(h0)
        else:
            self.register_buffer("log_h", h0)

    def _h2(self):
        h = self.log_h.exp()
        return (h * h).clamp_min(self.eps)

    def forward(self, x: torch.Tensor, guide: torch.Tensor | None = None):
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)

        if x.ndim != 5:
            raise ValueError(f"Expected x with shape (B, C, D, H, W), got {tuple(x.shape)}")

        if guide is None:
            guide = x
        if guide.shape != x.shape:
            raise ValueError(
                f"guide must have the same shape as x. Got {tuple(guide.shape)} vs {tuple(x.shape)}"
            )

        B, C, D, H, W = x.shape
        p = self.patch_size // 2
        s = self.search_size // 2
        total_pad = p + s

        # Pad once so every candidate patch/value is valid.
        x_pad = F.pad(
            x,
            (total_pad, total_pad, total_pad, total_pad, total_pad, total_pad),
            mode=self.pad_mode,
        )
        g_pad = F.pad(
            guide,
            (total_pad, total_pad, total_pad, total_pad, total_pad, total_pad),
            mode=self.pad_mode,
        )

        # Center crop for the reference location grid (size D,H,W).
        ref = g_pad[
            :,
            :,
            total_pad:total_pad + D,
            total_pad:total_pad + H,
            total_pad:total_pad + W,
        ]

        logits_list = []
        values_list = []

        h2 = self._h2()
        spatial_denom = 2.0 * (self.sigma_spatial ** 2) + self.eps

        for dz in range(-s, s + 1):
            for dy in range(-s, s + 1):
                for dx in range(-s, s + 1):
                    z0 = total_pad + dz
                    y0 = total_pad + dy
                    x0 = total_pad + dx

                    # Candidate crop aligned to the output grid.
                    cand_val = x_pad[:, :, z0:z0 + D, y0:y0 + H, x0:x0 + W]
                    cand_guide = g_pad[:, :, z0:z0 + D, y0:y0 + H, x0:x0 + W]

                    # Patch-wise mean squared difference.
                    diff2 = (ref - cand_guide).pow(2).mean(dim=1, keepdim=True)  # (B,1,D,H,W)
                    dist = F.avg_pool3d(
                        diff2,
                        kernel_size=self.patch_size,
                        stride=1,
                        padding=p,
                    ).squeeze(1)  # (B,D,H,W)

                    spatial_bias = -float(dz * dz + dy * dy + dx * dx) / spatial_denom
                    logits = -(dist / h2) + spatial_bias
                    logits_list.append(logits)
                    values_list.append(cand_val)

        # Softmax over all candidate offsets.
        logits = torch.stack(logits_list, dim=1)          # (B, K, D, H, W)
        weights = torch.softmax(logits, dim=1).unsqueeze(2)  # (B, K, 1, D, H, W)

        values = torch.stack(values_list, dim=1)          # (B, K, C, D, H, W)
        out = (weights * values).sum(dim=1)               # (B, C, D, H, W)

        return out