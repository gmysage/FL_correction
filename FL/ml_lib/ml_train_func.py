import torch
from .ml_loss import poisson_loss_image, gradient_loss
"""
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

"""

def run_validation_epoch(
    model,
    valid_loader,
    device,
    loss_r
):
    totals = {
        "total": 0.0,
        "mse": 0.0,
        "poisson": 0.0,
        "tv": 0.0
    }

    n_batches = 0

    for batch in valid_loader:
        out = valid_step_3DUNet(
            model,
            batch,
            device,
            loss_r
        )

        for k in totals:
            totals[k] += out[k]

        n_batches += 1

    for k in totals:
        totals[k] /= max(1, n_batches)

    return totals


def train_step_3DUNet(
    model,
    batch,
    optimizer,
    device,
    loss_r,
    grad_clip=1.0,
    lambda_consistency=0.1
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    noisy = batch["noisy"].to(device)   # (N, B, H, W)
    gt = batch["gt"].to(device)         # (N, B, H, W)

    # -------- reshape to (N, 1, D, H, W) --------
    noisy = noisy.unsqueeze(1)
    gt = gt.unsqueeze(1)

    B = noisy.shape[0]

    # -------- reference index (single-ref pretrain) --------
    ref_idx = torch.zeros(B, dtype=torch.long, device=device)

    # ======================================
    # Standard forward
    # ======================================
    pred = model(noisy, ref_idx)

    # ======================================
    # Transpose-consistency branch
    # ======================================

    # (N, 1, B, H, W)  ->  (N, 1, H, B, W)
    noisy_t = noisy.transpose(2, 3)

    pred_t = model(noisy_t, ref_idx)

    # transpose back to original orientation
    pred_t_back = pred_t.transpose(2, 3)

    # ======================================
    # Loss terms
    # ======================================
    loss_mse = torch.nn.functional.mse_loss(pred, gt)

    loss_poisson = poisson_loss_image(pred, gt)

    loss_tv = gradient_loss(pred)

    # --- NEW: consistency loss ---
    loss_consistency = torch.nn.functional.mse_loss(pred, pred_t_back)

    # ======================================
    # Total loss
    # ======================================
    loss = (
        loss_r['mse'] * loss_mse +
        loss_r['poisson'] * loss_poisson +
        loss_r['tv'] * loss_tv +
        lambda_consistency * loss_consistency
    )

    loss.backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return {
        "total": loss.item(),
        "mse": loss_mse.item(),
        "poisson": loss_poisson.item(),
        "tv": loss_tv.item(),
        "consistency": loss_consistency.item()
    }


###n validation step (single batch)


@torch.no_grad()
def valid_step_3DUNet(
    model,
    batch,
    device,
    loss_r
):
    model.eval()

    noisy = batch["noisy"].to(device)
    gt = batch["gt"].to(device)

    # reshape → (N, 1, D, H, W)
    noisy = noisy.unsqueeze(1)
    gt = gt.unsqueeze(1)

    B = noisy.shape[0]

    ref_idx = torch.zeros(B, dtype=torch.long, device=device)

    pred = model(noisy, ref_idx)

    loss_mse = torch.nn.functional.mse_loss(pred, gt)
    loss_poisson = poisson_loss_image(pred, gt)
    loss_tv = gradient_loss(pred)

    loss = loss_r['mse'] * loss_mse + \
           loss_r['poisson'] * loss_poisson + \
           loss_r['tv'] * loss_tv

    return {
        "total": loss.item(),
        "mse": loss_mse.item(),
        "poisson": loss_poisson.item(),
        "tv": loss_tv.item()
    }
