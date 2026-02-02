import torch
import torch.nn.functional as F

def torch_cal_xray_atten(
    C,            # (n_ref, H, W)
    C_other,      # (H, W)
    theta_list,   # (n_angle,)
    cs_ref,       # (n_angle, n_ref)
    rho,          # (H, W)
    pix
):
    device = C.device
    dtype = C.dtype

    n_ref, H, W = C.shape
    n_angle = theta_list.numel()

    C_sum = C.sum(dim=0) + C_other
    C_frac = C / (C_sum.max() + 1e-9)

    atten_xray = torch.empty((n_angle, H, W), device=device, dtype=dtype)

    for i in range(n_angle):
        frac_angle = torch_rotate_image_nd(C_frac, theta_list[i])  # must be grid_sample-based

        mu = torch.zeros((H, W), device=device, dtype=dtype)
        for j in range(n_ref):
            mu += cs_ref[i, j] * frac_angle[j] * rho * pix

        f = torch.clamp(1.0 - mu, min=1e-6)

        rev_f = torch.flip(f, dims=[0])
        rev_prod = torch.cumprod(rev_f, dim=0)

        body = torch.flip(rev_prod[:-1], dims=[0])
        ones = torch.ones((1, W), device=device, dtype=dtype)

        atten_xray[i] = torch.cat([body, ones], dim=0)

    return atten_xray




def torch_cal_xray_atten_vectorize(
    C,          # (n_ref, H, W)
    C_other,    # (H, W)
    theta,      # (n_angle,)
    cs_ref,     # (n_angle, n_ref)
    rho,        # (H, W)
    pix         # scalar
):
    """
    Fully vectorized, differentiable X-ray attenuation
    Returns: atten_xray (n_angle, H, W)
    """

    device = C.device
    dtype = C.dtype

    n_ref, H, W = C.shape
    n_angle = theta.numel()

    # ======================================================
    # Step 1: volume fraction
    # ======================================================
    C_sum = C.sum(dim=0) + C_other            # (H, W)
    C_frac = C / (C_sum.max() + 1e-8)         # (n_ref, H, W)

    # ======================================================
    # Step 2: rotate ALL angles at once
    # ======================================================
    # Expand to (n_angle, n_ref, H, W)
    C_frac = C_frac.unsqueeze(0).expand(n_angle, -1, -1, -1)

    # ---- build affine matrices ----
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    affine = torch.zeros((n_angle, 2, 3), device=device, dtype=dtype)
    affine[:, 0, 0] = cos_t
    affine[:, 0, 1] = -sin_t
    affine[:, 1, 0] = sin_t
    affine[:, 1, 1] = cos_t

    # ---- grid + sample ----
    grid = F.affine_grid(
        affine,
        size=(n_angle, 1, H, W),
        align_corners=True
    )
    # reshape for grid_sample: (N, C, H, W)
    frac_rot = F.grid_sample(
        C_frac.reshape(n_angle * n_ref, 1, H, W),
        grid.repeat_interleave(n_ref, dim=0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )
    frac_rot = frac_rot.reshape(n_angle, n_ref, H, W)
    # ======================================================
    # Step 3: compute μ(x, y, angle)
    # ======================================================
    # cs_ref: (n_angle, n_ref) → (n_angle, n_ref, 1, 1)
    mu = (frac_rot * cs_ref[:, :, None, None]).sum(dim=1) * rho * pix  # (n_angle, H, W)
    mu = torch.relu(mu)
    # ======================================================
    # Step 4: cumulative attenuation (NO Python loops)
    # ======================================================
    f = torch.clamp(1.0 - mu, min=1e-8)                            # (n_angle, H, W)

    # reverse row direction
    f_rev = torch.flip(f, dims=[1])            # flip H
    # cumulative product
    cumprod = torch.cumprod(f_rev, dim=1)
    # shift & restore
    atten_xray = torch.ones_like(f)
    atten_xray[:, :-1, :] = torch.flip(cumprod[:, 1:, :], dims=[1])

    return atten_xray


def torch_rotate_image_nd(
    x,          # (..., H, W)
    theta,      # scalar OR (...,) matching batch
    align_corners=True
):
    """
    Rotate images in x by theta (radians)

    Supports:
        (H, W)
        (N, H, W)
        (N1, N2, H, W)
        ...
    """

    device = x.device
    dtype = x.dtype

    *batch_dims, H, W = x.shape
    batch_size = int(torch.prod(torch.tensor(batch_dims))) if batch_dims else 1

    # --------------------------------------------------
    # reshape to (N, 1, H, W)
    # --------------------------------------------------
    x_flat = x.reshape(batch_size, 1, H, W)

    # --------------------------------------------------
    # theta handling
    # --------------------------------------------------
    if torch.numel(theta) == 1:
        theta = theta.expand(batch_size)
    else:
        theta = theta.reshape(-1)
        assert theta.numel() == batch_size, \
            "theta must match batch size"

    # --------------------------------------------------
    # affine matrices
    # --------------------------------------------------
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    affine = torch.zeros((batch_size, 2, 3), device=device, dtype=dtype)
    affine[:, 0, 0] = cos_t
    affine[:, 0, 1] = -sin_t
    affine[:, 1, 0] = sin_t
    affine[:, 1, 1] = cos_t

    # --------------------------------------------------
    # grid + sample
    # --------------------------------------------------
    grid = F.affine_grid(
        affine,
        size=(batch_size, 1, H, W),
        align_corners=align_corners
    )

    y_flat = F.grid_sample(
        x_flat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners
    )

    # --------------------------------------------------
    # reshape back
    # --------------------------------------------------
    y = y_flat.reshape(*batch_dims, H, W)
    return y