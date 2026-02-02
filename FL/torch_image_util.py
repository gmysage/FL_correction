import torch
import math
import torch.nn.functional as F

def _rotated_hw(H, W, theta):
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    H_new = int(H * cos_t + W * sin_t + 0.5)
    W_new = int(W * cos_t + H * sin_t + 0.5)
    return H_new, W_new


def torch_rotate_image_nd(
        x,
        theta,
        enable_crop=True,
        align_corners=False,
        mode="bilinear",
        padding_mode="zeros",
):
    """
    Rotate an N-D image tensor using PyTorch.

    Args
    ----
    x : torch.Tensor
        Shape (..., H, W)
    theta : float
        Rotation angle in radians (positive = CCW)
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
        y = y.squeeze(0)

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

