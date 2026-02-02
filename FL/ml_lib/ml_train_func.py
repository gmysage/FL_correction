import torch
from .ml_loss import poisson_loss_image, gradient_loss
def train_step_3DUNet(
    model,
    batch,
    optimizer,
    device,
    loss_r,
    grad_clip=1.0
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    noisy = batch["noisy"].to(device)  # (N, B, H, W)
    gt = batch["gt"].to(device)

    # -------- reshape to (N, 1, D, H, W) --------
    noisy = noisy.unsqueeze(1)
    gt = gt.unsqueeze(1)

    B = noisy.shape[0]

    # -------- reference index (single-ref pretrain) --------
    ref_idx = torch.zeros(B, dtype=torch.long, device=device)

    # -------- forward --------
    pred = model(noisy, ref_idx)

    # -------- loss --------
    loss_mse = torch.nn.functional.mse_loss(pred, gt)
    loss_poisson = poisson_loss_image(pred, gt)
    loss_tv = gradient_loss(pred)

    loss = loss_r['mse'] * loss_mse + \
            loss_r['poisson'] * loss_poisson + \
            loss_r['tv'] * loss_tv
    loss.backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return loss.item()

###n validation step (single batch)

@torch.no_grad()
def val_step_3DUNet(
    model,
    batch,
    device,
    loss_r
):
    model.eval()

    noisy = batch["noisy"].to(device).unsqueeze(1)
    gt = batch["gt"].to(device).unsqueeze(1)

    B = noisy.shape[0]
    ref_idx = torch.zeros(B, dtype=torch.long, device=device)

    pred = model(noisy, ref_idx)
    loss_mse = torch.nn.functional.mse_loss(pred, gt)
    loss_poisson = poisson_loss_image(pred, gt)
    loss_tv = gradient_loss(pred)

    loss = loss_r['mse'] * loss_mse + \
            loss_r['poisson'] * loss_poisson + \
            loss_r['tv'] * loss_tv

    return loss.item()