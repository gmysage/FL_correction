from .ml_dataloader import dataloader_3DCNN, get_gt_blur_dataloader
from .ml_model import RefAware3DUNet, RRDBNet
from .ml_model2 import RRDBNetMultiScaleResidual
from .ml_train_func import train_step_3DUNet, train_simple_denoise
from .ml_util import denoise_3d, load_default_3DUNet_model
from .ml_loss import extract_h_loss
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



def pre_train_3DUNet():
    device = 'cuda'
    gt_dir = '/data/FL_correction/FL_2/gt_image2'
    noisy_dir = '/data/FL_correction/FL_2/noisy_image2'
    model_3dunet = FL.load_default_3DUNet_model(None, device, start_freash=True, n_levels=4)
    model_3dunet = FL.train_3DUNet(gt_dir,
                   noisy_dir,
                   model_3dunet,
                   loss_r={'mse':1,'tv':1, 'consist':1, 'ssim':0},
                   epochs=30,  # Added epochs parameter to drive loop
                   lr=0.001,
                   batch_size=8,
                   split_ratio=0.8,
                   grad_clip=1,
                   device=device,
                   save_dir="/data/FL_correction/FL_2/saved_3dunet_model" # Added best model checkpoints
    )


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
    model = FL.RefAware3DUNet_general(
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


############## train 2D RRDB/UNet ##########
def example_train_rrdb_for_missing_angle_tomo():
    model_type = "multi_scale" # load this model: RRDBNetMultiScaleResidual
    # model_type = "regular" # load this model: RRDBNet
    device = 'cuda:0'
    start_fresh = False # load pre-trained model
    # start_fresh = True # training from scratch
    model_rrdb = FL.load_default_2DRRDB_model(model_type, device, start_fresh)
    lr = 0.001
    loss_r = {'mse':1, 'l1':1, 'tv':0}
    epochs = 20
    gt_dir = '/media/mingyuan/Data/dataset/recon_missing_angle/gt_image_random_rotate'
    noisy_dir = '/media/mingyuan/Data/dataset/recon_missing_angle/blur_image_random_rotate'

    """
    # refine training on shepp-logan shape
    gt_dir = '/media/mingyuan/Data/dataset/shepp_deform/shepp_deform_gt'
    noisy_dir = '/media/mingyuan/Data/dataset/shepp_deform/shepp_deform_blur_mlem'
    """
    model_save_dir = 'tmp_saved_model'
    model_save_name = 'best_rrdb_missing_angle_tomo.pth'
    model_rrdb, best_model_path = FL.train_rrdb(gt_dir,
                                                noisy_dir,
                                                model_rrdb,
                                                loss_r=loss_r,
                                                epochs=epochs,
                                                lr=lr,
                                                model_save_dir=model_save_dir,
                                                model_save_name=model_save_name,
                                                )
    # test on noisy image
    fn_noisy = np.sort(glob.glob(noisy_dir + '/*.tiff'))
    fn_clean = np.sort(glob.glob(gt_dir + '/*.tiff'))
    id_img = 200
    img_noisy = io.imread(fn_noisy[id_img])
    img_gt = io.imread(fn_clean[id_img])
    img_out = FL.apply_model(model_rrdb, img_noisy)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img_noisy)
    plt.title('noisy')
    plt.subplot(1, 3, 2)
    plt.imshow(img_out)
    plt.title('model output')
    plt.subplot(1, 3, 3)
    plt.imshow(img_gt)
    plt.title('clean')


def sample_train_simple_denoise():
    device = torch.device('cuda:2')
    lr = 0.001
    loss_r = {}
    loss_r['mse_img'] = 100
    loss_r['tv_img'] = 1
    loss_r['ssim_img'] = 1

    """
    loss_r['vgg'] = 0    
    global vgg19

    torch.manual_seed(0)
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()
    """

    h_loss_train = {}
    h_loss_valid = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value': [], 'rate': []}
        h_loss_valid[k] = {'value': [], 'rate': []}

    n_train = 10000
    batch_size = 32
    # in computer office2
    """
    f_root = '/data/denoise/RRDB'
    blur_dir = f_root + '/blur_images'
    gt_dir = f_root + '/gt_images'
    """
    f_root = '/media/mingyuan/Data/denoise/data_set'
    blur_dir = f_root + '/blur_tomo'
    gt_dir = f_root + '/gt_tomo'


    """
    # for missing angle training
    f_root = '/media/mingyuan/Data/dataset/recon_missing_angle'
    blur_dir = f_root + '/blur_image'
    gt_dir = f_root + '/gt_image'
    
    """
    #model_denoise = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_denoise = RRDBNetMultiScaleResidual(in_nc=1,
                                              out_nc=1,
                                              nf=16,
                                              nb=4,
                                              gc=32,
                                              lowres_rrdb=1,
                                              dilated_rrdb=2,
                                              dilation=2,
                                              residual_scale_init=0.2
                                              ).to(device)


    model_path = f_root + '/model_saved_vgg/model_0199.pth'
    # model_denoise.load_state_dict(torch.load(model_path, map_location=device))

    """
    # for missing angle tomo
    model_path = f_root + '/saved_rrdb_model/best_rrdb_model.pth'
    model_rrdb.load_state_dict(torch.load(model_path, map_location=device))
    model_save_dir = f_root + '/saved_model'
    model_rrdb, best_model_path = FL.train_rrdb(gt_dir,
                                                   blur_dir,
                                                   model_rrdb=model_denoise,
                                                   loss_r=loss_r,
                                                   epochs=100,  # Added epochs parameter to drive loop
                                                   lr=0.005,
                                                   batch_size=16,
                                                   split_ratio=0.8,
                                                   device=device,
                                                   save_dir=model_save_dir # Added best model checkpoints
                                    )

    """
    train_loader, valid_loader = get_gt_blur_dataloader(gt_dir, blur_dir, n_train, batch_size, 0.9,
                                                        exclude_range=[800, 24000], binning=2)

    epoch_s, epoch_e = 0, 300
    for epoch in range(epoch_s, epoch_e):
        print(f'epoch = {epoch}:')
        loss_summary_train, model_denoise = train_simple_denoise(train_loader, loss_r, model_denoise, lr, device=device)
        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(txt_t)

        model_folder = f'{f_root}/model_saved'
        ftmp = f'{model_folder}/model_{epoch:04d}.pth'
        if epoch % 10 == 1 or epoch == epoch_e - 1:
            torch.save(model_denoise.state_dict(), ftmp)
            with open(f'{model_folder}/h_loss_train_{epoch_s}_{epoch_e}.json', 'w') as f:
                json.dump(h_loss_train, f)