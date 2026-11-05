import os
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .ml_loss import (
    poisson_loss_image,
    psnr,
    ssim_loss,
    tv_loss_norm,
    vgg_loss,
    extract_h_loss,
    l1_loss
    )
from torch.nn import MSELoss
from tqdm import tqdm
from .ml_util import load_default_2DRRDB_model, load_default_3DUNet_model
from .ml_dataloader import (
    dataloader_RRDB_split,
    dataloader_3DCNN_split,
    save_img3D_to_2D_slice,
    save_img3D_to_3D_stack,
)
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
        out = valid_3DUNet_step(model ,batch, device, loss_r)
        for k in totals:
            totals[k] += out[k]
        n_batches += 1
    for k in totals:
        totals[k] /= max(1, n_batches)
    return totals


def train_3DUNet_step_obselete(
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

    noisy = batch["noisy"].to(device)   # (B, D, H, W) e.g., (8, 32, 200, 200)
    gt = batch["gt"].to(device)         # (B, D, H, W)

    # -------- reshape to (B, 1, D, H, W) --------
    noisy = noisy.unsqueeze(1)
    gt = gt.unsqueeze(1)

    B = noisy.shape[0]

    # -------- reference index (single-ref pretrain) --------
    ref_idx = torch.zeros(B, dtype=torch.long, device=device)

    pred = model(noisy, ref_idx)

    # ======================================
    # Transpose-consistency branch
    # ======================================
    # (B, 1, D, H, W)  ->  (B, 1, H, D, W)
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
def valid_3DUNet_step_obsolete(
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

"""
############ train 2D RRDB/UNet ########
def train_rrdb_step(model_rrdb, train_loader, loss_r, optimizer, vgg19=None, reshape_img=[0, 0]):
    mse_criterion = MSELoss()
    #optimizer = optim.Adam(model_rrdb.parameters(), lr=lr)
    model_rrdb.train()

    device = next(model_rrdb.parameters()).device

    keys = list(loss_r.keys())
    running_loss = {k: 0.0 for k in keys}
    running_psnr = 0.0

    for bi, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        label = data["gt"].to(device)
        image_data = data["noisy"].to(device)

        if not reshape_img == [0, 0]:
            new_shape = np.random.uniform(reshape_img[0], reshape_img[1])
            new_shape = int(new_shape // 4 * 4)
            new_shape = (new_shape, new_shape)
            label = F.interpolate(label, size=new_shape, mode='bilinear')
            image_data = F.interpolate(image_data, size=new_shape, mode='bilinear')



        output = model_rrdb(image_data)
        loss_value = {}

        for k in keys:
            if loss_r[k] <= 0:
                continue
            if 'mse' in k:
                loss_value[k] = mse_criterion(output, label)
            elif 'tv' in k:
                loss_value[k] = tv_loss_norm(output)
            elif 'ssim' in k:
                loss_value[k] = 1.0 - ssim_loss(output, label)
            elif 'poisson' in k:
                loss_value[k] = poisson_loss_image(output, label)
            elif 'l1' in k:
                loss_value[k] = l1_loss(output, label)
            elif 'vgg' in k and vgg19 is not None:
                loss_value[k] = vgg_loss(output, label, vgg19, device=device)

        total_loss_gen = sum([loss_value[k] * loss_r[k] for k in loss_value])
        optimizer.zero_grad(set_to_none=True)
        total_loss_gen.backward()
        optimizer.step()
        for k in keys:
            if k in loss_value:
                running_loss[k] += loss_value[k].item() * loss_r[k]
        batch_psnr = psnr(label, output)
        running_psnr += batch_psnr

    n_batches = len(train_loader)
    loss_summary = {k: running_loss[k] / n_batches for k in keys}
    loss_summary['psnr'] = running_psnr / n_batches

    return loss_summary, model_rrdb


@torch.no_grad()  # Disable gradient tracking to save huge memory and compute overhead
def validate_rrdb_step(model_rrdb, valid_loader, loss_r, vgg19=None, device=None):
    """
    Evaluates the model performance on the validation set without changing weights.
    """
    mse_criterion = MSELoss()
    model_rrdb.eval()  # Set model to evaluation mode (freezes BatchNorm/Dropout behavior)

    if device is None:
        device = next(model_rrdb.parameters()).device

    keys = list(loss_r.keys())
    running_loss = {k: 0.0 for k in keys}
    running_psnr = 0.0

    # Process validation batches (Disable progress bar nested clutter using leave=False)
    for data in tqdm(valid_loader, desc="Validating", leave=False):
        label = data['gt'].to(device)
        image_data = data['noisy'].to(device)

        output = model_rrdb(image_data)
        loss_value = {}

        # Evaluate each active loss component
        for k in keys:
            if loss_r[k] <= 0:
                continue

            if 'mse' in k:
                loss_value[k] = mse_criterion(output, label)
            elif 'tv' in k:
                loss_value[k] = tv_loss_norm(output)
            elif 'ssim' in k:
                loss_value[k] = 1.0 - ssim_loss(output, label)
            elif 'poisson' in k:
                loss_value[k] = poisson_loss_image(output, label)
            elif 'l1' in k:
                loss_value[k] = l1_loss(output, label)
            elif 'vgg' in k and vgg19 is not None:
                loss_value[k] = vgg_loss(output, label, vgg19, device=device)

        # Accumulate metrics
        for k in keys:
            if k in loss_value:
                running_loss[k] += loss_value[k].item() * loss_r[k]

        batch_psnr = psnr(label, output)
        running_psnr += batch_psnr

    # Normalize metrics across validation sets
    n_batches = len(valid_loader)
    val_summary = {}
    for k in keys:
        val_summary[k] = running_loss[k] / n_batches
    val_summary['psnr'] = running_psnr / n_batches

    return val_summary


def train_rrdb(gt_dir,
               noisy_dir,
               model_rrdb,
               loss_r=None,
               epochs=10,  # Added epochs parameter to drive loop
               lr=0.001,
               batch_size=16,
               split_ratio=0.8,
               model_save_dir="./saved_rrdb_model", # Added best model checkpoints
               model_save_name="best_rrdb_model.pth",
               reshape_img=[0, 0]
):
    device = next(model_rrdb.parameters()).device
    os.makedirs(model_save_dir, exist_ok=True)
    if loss_r is None:
        loss_r = {'mse': 1.0}
    keys = list(loss_r.keys())
    h_loss_train = {}
    h_loss_valid = {}
    for k in keys:
        h_loss_train[k] = {'value': [], 'rate': []}
        h_loss_valid[k] = {'value': [], 'rate': []}

    train_loader, valid_loader = dataloader_RRDB_split(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        batch_size=batch_size,
        split_ratio=split_ratio,
    )
    vgg19 = None
    if any('vgg' in k and loss_r[k] > 0 for k in loss_r.keys()):
        weights = torchvision.models.VGG19_Weights.DEFAULT
        vgg19 = torchvision.models.vgg19(weights=weights).features
        for param in vgg19.parameters():
            param.requires_grad_(False)
        vgg19.to(device).eval()

    best_psnr = float('-inf')
    if not '.pth' in model_save_name:
        model_save_name = model_save_name + '.pth'
    best_model_path = os.path.join(model_save_dir, model_save_name)
    optimizer = optim.Adam(model_rrdb.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_summary, model_rrdb = train_rrdb_step(model_rrdb,
                                                    train_loader,
                                                    loss_r,
                                                    optimizer,
                                                    vgg19,
                                                    reshape_img=reshape_img
                                                    )
        h_loss_train, txt_train, psnr_train = extract_h_loss(h_loss_train, train_summary, loss_r)
        print(txt_train)
        #print(f"[Train] Loss:  " + ", ".join([f"{k}: {v:.4e}" for k, v in train_summary.items()]))

        if len(valid_loader):
            val_summary = validate_rrdb_step(model_rrdb, valid_loader, loss_r, vgg19, device)
            h_loss_valid, txt_valid, psnr_valid = extract_h_loss(h_loss_valid, val_summary, loss_r)
            print(txt_valid)
            current_psnr = psnr_valid
        else:
            current_psnr = psnr_train
            #current_psnr = train_summary.get('psnr', float('-inf'))

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            checkpoint = {
                'model_config': model_rrdb.model_config,
                'epoch': epoch,
                'model_state_dict': model_rrdb.state_dict(),
                'best_psnr': best_psnr,
                'loss_summary': train_summary,
                'train_loss_summary': h_loss_train,
                'valid_loss_summary': h_loss_valid,
            }
            torch.save(checkpoint, best_model_path)
            print(f"Best PSNR ({best_psnr:.2f} dB), saved checkpoint to: {best_model_path}")

    print(f"\nFolder training completed. Best recorded Validation PSNR: {best_psnr:.2f} dB")
    return model_rrdb, best_model_path


def train_rrdb_image_stack(img3D_gt,
                           img3D_blur,
                           model_rrdb=None,
                           loss_r=None,
                           epochs=10,
                           lr=0.005,
                           batch_size=16,
                           split_ratio=1,
                           device='cuda',
                           save_dir="./saved_rrdb_model"
                           ):

    gt_dir, noisy_dir = save_img3D_to_2D_slice(img3D_gt, img3D_blur)
    model_rrdb, best_model_path = train_rrdb(gt_dir,
                                               noisy_dir,
                                               model_rrdb,
                                               loss_r,
                                               epochs,
                                               lr,
                                               batch_size,
                                               split_ratio,
                                               device,
                                               save_dir
                                            )
    return model_rrdb, best_model_path


############################
def train_3DUNet(gt_dir,
                 noisy_dir,
                 model_3dunet=None,
                 loss_r=None,
                 epochs=10,
                 lr=0.005,
                 batch_size=16,
                 split_ratio=0.8,
                 grad_clip=None,
                 device='cuda',
                 save_dir="./saved_3dunet_model"):
    os.makedirs(save_dir, exist_ok=True)

    if loss_r is None:
        loss_r = {'mse': 1.0, 'consist': 1.0, 'tv': 1e-2}
        print(f'set default loss r = {loss_r}\n')

    keys = list(loss_r.keys())
    h_loss_train = {k: {'value': [], 'rate': []} for k in keys}
    h_loss_valid = {k: {'value': [], 'rate': []} for k in keys}

    if model_3dunet is None:
        model_3dunet = load_default_3DUNet_model(id_model=None, device=device)

    train_loader, valid_loader = dataloader_3DCNN_split(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        batch_size=batch_size,
        split_ratio=split_ratio,
    )

    optimizer = optim.Adam(model_3dunet.parameters(), lr=lr)

    best_psnr = float('-inf')
    best_model_path = os.path.join(save_dir, "best_3dunet_model.pth")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_summary, model_3dunet = train_3DUNet_step(
            model_3dunet,
            train_loader,
            loss_r,
            optimizer,
            grad_clip=grad_clip,
            device=device
        )
        h_loss_train, txt_train, psnr_train = extract_h_loss(h_loss_train, train_summary, loss_r)
        print(txt_train)

        if len(valid_loader):
            val_summary = valid_3DUNet_step(model_3dunet, valid_loader, loss_r, device)
            h_loss_valid, txt_valid, psnr_valid = extract_h_loss(h_loss_valid, val_summary, loss_r)
            print(txt_valid)
            current_psnr = psnr_valid
        else:
            current_psnr = psnr_train

        is_best = current_psnr > best_psnr
        if is_best:
            best_psnr = current_psnr

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_3dunet.state_dict(),
            'best_psnr': best_psnr,
            'current_psnr': current_psnr,
            'loss_summary': train_summary
        }

        epoch_model_path = os.path.join(save_dir, f"model_3dunet_epoch_{epoch:04d}.pth")
        torch.save(checkpoint, epoch_model_path)

        if is_best:
            torch.save(checkpoint, best_model_path)
            print(f"Best PSNR ({best_psnr:.2f} dB), saved checkpoint to: {best_model_path}")

    print(f"\nFolder training completed. Best recorded Validation PSNR: {best_psnr:.2f} dB")
    return model_3dunet, best_model_path


def train_3dunet_image_stack(img3D_gt,
                           img3D_blur,
                           model_3dunet=None,
                           loss_r=None,
                           epochs=10,
                           lr=0.005,
                           batch_size=16,
                           split_ratio=1,
                           n_sli=32,
                           device='cuda',
                           save_dir="./saved_3dunet_model"
                           ):

    gt_dir, noisy_dir = save_img3D_to_3D_stack(img3D_gt, img3D_blur, n_sli)
    model_3dunet_r, best_model_path = train_3DUNet(gt_dir,
                                                   noisy_dir,
                                                   model_3dunet,
                                                   loss_r,
                                                   epochs,
                                                   lr,
                                                   batch_size,
                                                   split_ratio,
                                                   grad_clip=None,
                                                   device=device,
                                                   save_dir=save_dir
)
    return model_3dunet_r, best_model_path


def train_3DUNet_step(model_3dunet, train_loader, loss_r, optimizer, grad_clip=None, device=None):
    mse_criterion = MSELoss()
    model_3dunet.train()

    if device is None:
        device = next(model_3dunet.parameters()).device

    keys = list(loss_r.keys())
    running_loss = {k: 0.0 for k in keys}
    running_psnr = 0.0

    for bi, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        gt = data["gt"].to(device)
        noisy = data["noisy"].to(device)

        noisy = noisy.unsqueeze(1)
        gt = gt.unsqueeze(1)

        B = noisy.shape[0]
        ref_idx = torch.zeros(B, dtype=torch.long, device=device)

        pred = model_3dunet(noisy, ref_idx)

        noisy_t = noisy.transpose(2, 3)
        pred_t = model_3dunet(noisy_t, ref_idx)
        pred_t_back = pred_t.transpose(2, 3)

        loss_value = {}

        for k in keys:
            if loss_r[k] <= 0:
                continue
            if 'mse' in k:
                loss_value[k] = mse_criterion(pred, gt)
            elif 'tv' in k:
                loss_value[k] = tv_loss_norm(pred)
            elif 'ssim' in k:
                pred_s = pred.squeeze(1)
                gt_s = gt.squeeze(1)
                loss_value[k] = 1.0 - ssim_loss(pred_s, gt_s)
            elif 'poisson' in k:
                loss_value[k] = poisson_loss_image(pred, gt)
            elif 'consist' in k:
                loss_value[k] = mse_criterion(pred, pred_t_back)

        if len(loss_value) == 0:
            raise RuntimeError("No active loss terms were found. Check loss_r keys and weights.")

        total_loss_gen = sum(loss_value[k] * loss_r[k] for k in loss_value)

        optimizer.zero_grad(set_to_none=True)
        total_loss_gen.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model_3dunet.parameters(), grad_clip)

        optimizer.step()

        for k in keys:
            if k in loss_value:
                running_loss[k] += loss_value[k].item() * loss_r[k]

        batch_psnr = psnr(gt, pred)
        running_psnr += batch_psnr

    n_batches = len(train_loader)
    loss_summary = {k: running_loss[k] / n_batches for k in keys}
    loss_summary['psnr'] = running_psnr / n_batches

    return loss_summary, model_3dunet


@torch.no_grad()
def valid_3DUNet_step(
    model_3dunet,
    valid_loader,
    loss_r,
    device,
):
    mse_criterion = MSELoss()
    model_3dunet.eval()
    if device is None:
        device = next(model_3dunet.parameters()).device
    keys = list(loss_r.keys())
    running_loss = {k: 0.0 for k in keys}
    running_psnr = 0.0
    for bi, data in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Training"):
        noisy = data["noisy"].to(device)
        gt = data["gt"].to(device)

        # reshape → (N, 1, D, H, W)
        noisy = noisy.unsqueeze(1)
        gt = gt.unsqueeze(1)
        B = noisy.shape[0]
        ref_idx = torch.zeros(B, dtype=torch.long, device=device)

        noisy_t = noisy.transpose(2, 3)
        pred_t = model_3dunet(noisy_t, ref_idx)
        # transpose back to original orientation
        pred_t_back = pred_t.transpose(2, 3)

        pred = model_3dunet(noisy, ref_idx)

        loss_value = {}

        # Evaluate each active loss component
        for k in keys:
            if loss_r[k] <= 0:
                continue
            if 'mse' in k:
                loss_value[k] = mse_criterion(pred, gt)
            elif 'tv' in k:
                loss_value[k] = tv_loss_norm(pred)
            elif 'ssim' in k:
                loss_value[k] = 1.0 - ssim_loss(pred.squeeze(), gt.squeeze())  # (B, H, D, W)
            elif 'poisson' in k:
                loss_value[k] = poisson_loss_image(pred, gt)
            elif 'consist' in k:
                loss_value[k] = mse_criterion(pred, pred_t_back)
        # Accumulate metrics
        for k in keys:
            if k in loss_value:
                running_loss[k] += loss_value[k].item() * loss_r[k]

        batch_psnr = psnr(gt, pred)
        running_psnr += batch_psnr

    # Normalize metrics across validation sets
    n_batches = len(valid_loader)
    val_summary = {}
    for k in keys:
        val_summary[k] = running_loss[k] / n_batches
    val_summary['psnr'] = running_psnr / n_batches

    return val_summary













############################



def forward_stack_without_grad(model, img3D, batch_size=1, device=None):
    """
    Forward a 3D stack through a 2D model in mini-batches.
    Returns a tensor with autograd history preserved.
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()
    if torch.is_tensor(img3D):
        img3D_tensor = img3D.to(device)
    else:
        img3D_tensor = torch.as_tensor(img3D, dtype=torch.float32).to(device)
    n_slices = img3D_tensor.shape[0]
    outs = []
    with torch.no_grad():
        for start in range(0, n_slices, batch_size):
            end = min(start + batch_size, n_slices)
            x = img3D_tensor[start:end].unsqueeze(1)   # (B, 1, H, W)
            y = model(x)                                          # graph is kept
            outs.append(y)
    out_stack = torch.cat(outs, dim=0)                            # (N, 1, H, W)
    return out_stack








def forward_stack_with_grad(model, img3D, batch_size=1, device=None):
    """
    Forward a 3D stack through a 2D model in mini-batches.
    Returns a tensor with autograd history preserved.
    """

    device = next(model.parameters()).device
    img3D_tensor = torch.as_tensor(img3D, dtype=torch.float32)
    n_slices = img3D_tensor.shape[0]
    outs = []
    for start in range(0, n_slices, batch_size):
        end = min(start + batch_size, n_slices)
        x = img3D_tensor[start:end].unsqueeze(1).to(device)   # (B, 1, H, W)
        y = model(x)                                   # graph is kept
        outs.append(y)
    out_stack = torch.cat(outs, dim=0)                  # (N, 1, H, W)
    return out_stack


def refine_on_stack(model, optimizer, img3D, img3D_gt, batch_size=8, device="cuda"):
    model.train()

    img3D_gt = torch.as_tensor(img3D_gt, dtype=torch.float32)

    out_stack = forward_stack_with_grad(model, img3D, batch_size=batch_size, device=device)

    y_true = img3D_gt.unsqueeze(1).to(device)          # (N, 1, H, W)
    loss =torch.nn.functional.mse_loss(out_stack, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return out_stack, loss.item()