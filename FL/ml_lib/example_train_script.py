from .ml_dataloader import dataloader_3DCNN
from .ml_model import RefAware3DUNet
from .ml_train_func import train_step_3DUNet
from .ml_util import denoise_3d, load_default_3DUNet_model
from tqdm import tqdm, trange
from skimage import io
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import json
import FL
import pyxas
from datetime import datetime
def pre_train_3DUNet_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fsave_model_root = '/data/FL_correction/FL_2/saved_model_UNet_4'
    # ---------------------------------
    # Hyperparameters
    # ---------------------------------
    n_epochs = 50
    batch_size = 8
    lr = 1e-4
    grad_clip = 1.0

    train_loader = FL.dataloader_3DCNN(
        "/data/FL_correction/FL_2/gt_image",
        "/data/FL_correction/FL_2/noisy_image",  # "/data/FL_correction/FL_2/noisy_image2",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    """
    model = FL.RefAware3DUNet(
        n_ref=1,  # single-reference pretraining
        emb_dim=8,
        base_ch=32
    ).to(device)
    """

    model = FL.RefAware3DUNet_g(
        n_ref=1,  # single-reference pretraining
        emb_dim=8,
        base_ch=32,
        n_levels=4
    ).to(device)

    model_load_path = fsave_model_root + f'/refaware3dunet_{199:04d}.pth'
    model.load_state_dict(torch.load(model_load_path, map_location=device))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # ---------------------------------
    # Loss history
    # ---------------------------------
    train_losses = []
    val_losses = []
    loss_r = {}
    loss_r['mse'] = 1
    loss_r['poisson'] = 0.1
    loss_r['tv'] = 0.1

    # ---------------------------------
    # Training loop
    # ---------------------------------
    ts = time.time()
    for epoch in range(100):
        # print(f\n\n'epoch = {epoch}')
        # -------- Training --------
        epoch_train_loss = 0.0
        n_train_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            leave=True
        )

        for batch in pbar:
            loss = FL.train_step_3DUNet(
                model,
                batch,
                optimizer,
                device,
                loss_r,
                grad_clip=grad_clip
            )

            epoch_train_loss += loss
            n_train_batches += 1

        epoch_train_loss /= max(1, n_train_batches)
        train_losses.append(epoch_train_loss)

        """
        # -------- Validation --------
        epoch_val_loss = 0.0
        n_val_batches = 0

        for batch in val_loader:
            loss = val_step_3DCNN(
                model,
                batch,
                device,
                alpha=alpha_loss
            )

            epoch_val_loss += loss
            n_val_batches += 1

        epoch_val_loss /= max(1, n_val_batches)
        val_losses.append(epoch_val_loss)
        """

        # -------- Logging --------
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(
                f"[Epoch {epoch + 1:03d}/{n_epochs}] "
                f"Train Loss: {epoch_train_loss:.4e} | "
            )
        # save model
        model_path = fsave_model_root + f'/refaware3dunet_{200 + epoch:04d}.pth'
        torch.save(model.state_dict(), model_path)
    # ---------------------------------
    # Convert loss history to numpy
    # ---------------------------------
    # train_losses = np.array(train_losses)
    # val_losses = np.array(val_losses)

    print("\nTraining finished.")
    print(f"Final Train Loss: {train_losses[-1]:.4e}")
    plt.figure()
    plt.plot(train_losses)
    te = time.time()
    print(f'trainging takes {te-ts} seconds')


##############################################################
########### training with self-consistance loss (permute axis)
##############################################################


def pre_train_3DUNet_self_consistance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fsave_model_root = '/data/FL_correction/FL_2/saved_model_UNet_4'
    # ---------------------------------
    # Hyperparameters
    # ---------------------------------
    n_epochs = 50
    batch_size = 4
    lr = 1e-4
    grad_clip = 1.0

    train_loader, valid_loader = FL.dataloader_3DCNN_split(
        gt_dir="/data/FL_correction/FL_2/gt_image",
        noisy_dir="/data/FL_correction/FL_2/noisy_image",
        batch_size=batch_size,
        train_ratio=0.6,
        num_workers=0
    )

    """
    train_loader = FL.dataloader_3DCNN(
        "/data/FL_correction/FL_2/gt_image",
        "/data/FL_correction/FL_2/noisy_image",  # "/data/FL_correction/FL_2/noisy_image2",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    """
    model = FL.RefAware3DUNet_g(
        n_ref=1,  # single-reference pretraining
        emb_dim=8,
        base_ch=32,
        n_levels=4
    ).to(device)

    model_load_path = fsave_model_root + f'/refaware3dunet_{365:04d}.pth'
    model.load_state_dict(torch.load(model_load_path, map_location=device))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_r = {}
    loss_r['mse'] = 1
    loss_r['poisson'] = 0#1e-3
    loss_r['tv'] = 0.01
    lambda_consistency = 20

    # ---------------------------------
    # Training loop
    # ---------------------------------
    train_log = []
    valid_log = []
    ts = time.time()
    for epoch in range(52):

        # -------- Training --------
        epoch_stats = {
            "total": 0.0,
            "mse": 0.0,
            "poisson": 0.0,
            "tv": 0.0,
            "consistency": 0.0
        }

        n_train_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            leave=True
        )

        for batch in pbar:
            epoch_train = FL.train_step_3DUNet(
                model,
                batch,
                optimizer,
                device,
                loss_r,
                grad_clip=grad_clip,
                lambda_consistency=lambda_consistency
            )

            # accumulate each term
            for k in epoch_stats.keys():
                epoch_stats[k] += epoch_train[k]

            n_train_batches += 1

            # update tqdm display
            pbar.set_postfix({
                "loss": f"{epoch_train['total']:.3e}",
                "mse": f"{epoch_train['mse']:.2e}",
                "cons": f"{epoch_train['consistency']:.2e}"
            })

        # -------- Average over batches --------
        for k in epoch_stats.keys():
            epoch_stats[k] /= max(1, n_train_batches)

        train_log.append(epoch_stats)

        # ===== VALIDATION =====
        epoch_valid = FL.run_validation_epoch(
            model,
            valid_loader,
            device,
            loss_r
        )

        valid_log.append(epoch_valid)

        # -------- Logging --------
        print(
            f"[Epoch {epoch + 1:03d}/{n_epochs}]\n "
            f"TRAIN: total={epoch_train['total']:.4e}, "
            f"mse={epoch_train['mse']:.4e}, "
            f"poisson={epoch_train['poisson']:.4e}, "
            f"tv={epoch_train['tv']:.4e},  "
            f"consistency={epoch_train['consistency']:.4e}\n "
            
            
            f"VALID: total={epoch_valid['total']:.4e}, "
            f"mse={epoch_valid['mse']:.4e}, "
            f"poisson={epoch_valid['poisson']:.4e}, "
            #f"consistency={epoch_valid['consistency']:.4e}, "
            f"tv={epoch_valid['tv']:.4e}\n"
        )
        # -------- Save model --------
        model_path = fsave_model_root + f'/refaware3dunet_{249+epoch:04d}.pth'
        torch.save(model.state_dict(), model_path)

        with open(fsave_model_root + "/training_log.json", "w") as f:
            json.dump(train_log, f, indent=2)
    print("\nTraining finished.")
    te = time.time()
    print(f'training takes {te-ts:.2f} seconds')
    print(datetime.fromtimestamp(te))

def plot_log(log):
    keys = log[0].keys()
    n = len(keys)
    n_r = int(np.sqrt(n))
    n_c = int(np.ceil(n/n_r))
    log_k = {}
    plt.figure()
    j = 1
    for k in keys:
        log_k[k] = [log[i][k] for i in range(len(log))]
        plt.subplot(n_r, n_c, j)
        plt.plot(log_k[k], label=k)
        plt.legend()
        j += 1
    return log_k

# inference example
def inferece_example():
    #id_model = 48  # this is best overall
    id_model = None
    device = 'cuda'
    model = load_default_3DUNet_model(id_model, device)

    fn_noisy = np.sort(glob.glob('/data/FL_correction/FL_2/noisy_image/*.tiff'))
    fn_gt = np.sort(glob.glob('/data/FL_correction/FL_2/gt_image/*.tiff'))
    img_c = []
    for k in trange(len(fn_noisy)):
        img_n = io.imread(fn_noisy[k])
        img_g = io.imread(fn_gt[k])
        # Example input
        # volume shape: (B, H, W)
        denoised = FL.denoise_3d_v0(
            volume_3d=img_n,
            model=model,
            device=device,
        )
        a = np.hstack((img_g[0], img_n[0], denoised[0]))
        img_c.append(a)
    img_c = np.array(img_c)
    #io.imsave('/data/FL_correction/FL_2/img_comp/img_comp_validate.tiff', img_c.astype(np.float32))

    print(denoised.shape)  #
    return denoised, img_c