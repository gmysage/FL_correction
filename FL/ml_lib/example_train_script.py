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


def pre_train_3DUNet_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fsave_model_root = '/data/FL_correction/FL_2/saved_model_UNet_gridrec'
    # ---------------------------------
    # Hyperparameters
    # ---------------------------------
    n_epochs = 50
    batch_size = 2
    lr = 1e-4
    grad_clip = 1.0

    train_loader = dataloader_3DCNN(
        "/data/FL_correction/FL_2/gt_image2",
        "/data/FL_correction/FL_2/noisy_image2_gridrec",  # "/data/FL_correction/FL_2/noisy_image2",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model = RefAware3DUNet(
        n_ref=1,  # single-reference pretraining
        emb_dim=8,
        base_ch=32
    ).to(device)

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
    for epoch in range(50, 100):
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
            loss = train_step_3DUNet(
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
        model_path = fsave_model_root + f'/refaware3dunet_{epoch:04d}.pth'
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


# inference example
def inferece_example():
    #id_model = 48  # this is best overall
    id_model = None
    device = 'cuda'
    model = load_default_3DUNet_model(id_model, device)

    fn_noisy = np.sort(glob.glob('/data/FL_correction/FL_2/noisy_image2/*.tiff'))
    fn_gt = np.sort(glob.glob('/data/FL_correction/FL_2/gt_image2/*.tiff'))
    img_c = []
    for k in trange(len(fn_noisy)):
        img_n = io.imread(fn_noisy[k])
        img_g = io.imread(fn_gt[k])
        # Example input
        # volume shape: (B, H, W)
        denoised = denoise_3d(
            volume_3d=img_n,
            model=model,
            device="cuda",
        )
        a = np.hstack((img_g[0], img_n[0], denoised[0]))
        img_c.append(a)
    img_c = np.array(img_c)
    #io.imsave('/data/FL_correction/FL_2/img_comp/img_comp_validate.tiff', img_c.astype(np.float32))

    print(denoised.shape)  #
    return denoised, img_c