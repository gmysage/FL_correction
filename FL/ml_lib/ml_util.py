import numpy as np
import os
import glob
import torch
import math
import importlib.util
import matplotlib.pyplot as plt
from skimage.transform import resize
from .ml_model import RefAware3DUNet, RRDBNet, RefAware3DUNet_general
from .ml_model2 import RRDBNetMultiScaleResidual
from tqdm import tqdm, trange


def load_checkpoint(model, checkpoint_path, device=None):
    if device is None:
        device = next(model.parameters()).device
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as err:
            print('fail to load checkpoint')
            print(err)
            return None
    return model



def load_default_3DUNet_model(id_model=None, device='cuda', start_freash=False, n_levels=4):
    spec = importlib.util.find_spec("FL")
    fn_model_root = spec.submodule_search_locations[0] # e.g. '/data/FL_correction/FL'

    if id_model is None:
        model_path = fn_model_root + '/ml_lib/saved_model/best_3dunet_model.pth'
        #model_path = fn_model_root + '/ml_lib/saved_model/refaware3dUnet_0263.pth'
    else:
        fn_model = fn_model_root + '/ml_lib/saved_model/saved_model_UNet'
        model_list = np.sort(glob.glob(fn_model + '/*.pth'))
        model_path = model_list[id_model]
    model_3DUNet = RefAware3DUNet_general(
            n_ref=1,        # single-reference pretraining
            emb_dim=8,
            base_ch=32,
            n_levels=n_levels
            )
    if not start_freash:
        try:
            model_3DUNet.load_state_dict(torch.load(model_path, map_location=device))
        except:
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model_3DUNet.load_state_dict(checkpoint['model_state_dict'])
            except Exception as err:
                print('fail to load default model')
                print(err)
                return None

    model_3DUNet.to(device)
    return model_3DUNet


@torch.no_grad()
def denoise_3d(
        volume_3d,  # numpy array: (B, H, W) or (n_ref, B, H, W)
        model,
        l=200,  # patch size
        Dl=50,  # overlap size
        device="cuda",
):
    '''
    limited by GPU memory, will do 3D denoising part-by-part (100, 100, 100)
    '''
    model = model.to(device)
    model.eval()
    ceil = math.ceil
    s = volume_3d.shape
    if len(s) == 3:
        img4D = volume_3d[np.newaxis]
    else:
        img4D = volume_3d

    img4D_cuda = torch.from_numpy(img4D).float().to(device)
    scale = torch.max(img4D_cuda) * 1.2
    img4D_cuda = img4D_cuda / scale

    n_ref, B, H, W = img4D_cuda.shape

    ref_idx = torch.zeros(n_ref, dtype=torch.long, device=device)
    tl = l - Dl

    num = int(ceil(B / tl) * ceil(H / tl) * ceil(W / tl))
    dl = int(Dl / 2)

    bs, hs, ws = 0, 0, 0
    be, he, we = l, l, l
    img_d = torch.zeros_like(img4D_cuda)
    with tqdm(total=num, desc='3D denoising') as pbar:
        while bs < B - 1:
            hs = 0
            he = hs + l
            while hs < H - 1:
                ws = 0
                we = ws + l
                while ws < W - 1:
                    data = img4D_cuda[:, bs:be, hs:he, ws:we]
                    noisy = data.unsqueeze(1)
                    pred = model(noisy, ref_idx).detach().squeeze(1)
                    w1 = ws + dl if ws > 0 else 0
                    dw = dl if ws > 0 else 0

                    h1 = hs + dl if hs > 0 else 0
                    dh = dl if hs > 0 else 0

                    b1 = bs + dl if bs > 0 else 0
                    db = dl if bs > 0 else 0

                    img_d[:, b1:be, h1:he, w1:we] = pred[:, db:, dh:, dw:]
                    pbar.update(1)
                    ws = we - Dl
                    we = ws + l
                hs = he - Dl
                he = hs + l
            bs = be - Dl
            be = bs + l

    img_d = img_d * scale
    img_d[img_d < 0] = 0
    img_d = img_d.cpu().numpy()
    img_d = img_d.squeeze()
    return img_d


@torch.no_grad()
def denoise_3d_v0(
        volume_3d,  # (B, H, W) or (n_ref, B, H, W)
        model,
        device="cuda",
):
    model = model.to(device)
    model.eval()
    if not torch.is_tensor(volume_3d):
        volume_3d = torch.from_numpy(volume_3d).float()

    s = volume_3d.shape
    if len(s) == 3:  # s = (B, H, W)
        img4D = volume_3d.unsqueeze(0).to(device)
    else:  # s = (n_ref, B, H, W)
        img4D = volume_3d.to(device)

    scale = torch.max(img4D) * 1.2
    img4D_n = img4D / scale
    noisy = img4D_n.unsqueeze(1)
    B = noisy.shape[0]
    ref_idx = torch.zeros(B, dtype=torch.long, device=device)

    pred = model(noisy, ref_idx).detach()
    pred = pred * scale
    img_denoise = pred.cpu().numpy().squeeze()

    img_denoise[img_denoise < 0] = 0
    return img_denoise

#############################################
def load_model_and_checkpoint(checkpoint_path, device='cuda'):
    """
    checkpoint should have the argument of "model_config"
    the model archtecture is defined in model_config['model_structure']
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']

    kwargs = {}
    for key, val in model_config.items():
        if key == 'model_structure':
            continue
        kwargs[key] = val

    func = globals()[model_config['model_structure']]
    model_rrdb = func(**kwargs)
    model_rrdb.load_state_dict(checkpoint['model_state_dict'])

    return model_rrdb


def load_default_2DRRDB_model(model_type='regular', device='cuda', start_fresh=False):
    """
    model_type:
        'regular': conventional RRDB model defined in ml_lib/ml_model.py
        'multi_scale': RRDBNetMultiScaleResidual RRDB model defined in ml_lib/ml_model2.py
                       trained for missing-angle tomo correction
    """
    spec = importlib.util.find_spec("FL")
    fn_model_root = spec.submodule_search_locations[0] # e.g. '/data/FL_correction/FL'

    if 'reg' in model_type:
        model_sub_path = '/ml_lib/saved_model/regular_rrdb_model.pth'
        model_rrdb = RRDBNet(1, 1, 16, 4, 32).to(device)
    elif 'multi' in model_type:
        model_sub_path = '/ml_lib/saved_model/multi_scale_rrdb_missing_angle.pth'
        model_rrdb = RRDBNetMultiScaleResidual(in_nc=1,
                                               nf = 16,
                                               nb = 4,
                                               gc = 32,
                                               lowres_rrdb = 3,#1
                                               dilated_rrdb = 3,#2
                                               dilation = 3,#2
                                               residual_scale_init = 0.2
                                               ).to(device)
    else:
        print('un-recognized model_type: choose from "regular" or "multi_scale"')
        return None
    model_path = fn_model_root + model_sub_path
    if not start_fresh:
        model_rrdb = load_checkpoint(model_rrdb, model_path)
    model_rrdb.to(device)
    return model_rrdb


def denoise_2d_RRDB(image, # 2D or 3D stack
                    model=None,
                    device='cuda',
                    plot_flag=0,
                    clim=None,
                    ax='off',
                    title='',
                    figsize=(14, 8)):
    if ax == 'off':
        axes = 'off'
    else:
        axes = 'on'
    if len(image.shape) == 2:
        img = np.expand_dims(image, 0)
    else:
        img = image.copy()
    n = len(img)
    if model is None:
        model = load_default_2DRRDB_model(device)
    else:
        device = next(model.parameters()).device
    img_torch = torch.tensor(img, dtype=torch.float).to(device)
    img_denoise = torch.zeros_like(img_torch)
    #model = model.to(device)
    with torch.no_grad():
        for i in trange(n):
            img_i = img_torch[i:i+1].unsqueeze(1)
            img_fit = model(img_i)
            img_fit = torch.clamp(img_fit, 0)
            img_denoise[i] = img_fit.squeeze()
    if plot_flag:
        plt.figure(figsize=figsize)
        if clim is None:
            clim = [np.min(image), np.max(image)]
        img0 = img[n//2].squeeze()
        img1 = img_denoise[n//2].detach().cpu().numpy().squeeze()
        plt.subplot(121);plt.imshow(img0, clim=clim)
        plt.axis(axes);plt.title('raw image');plt.colorbar()
        plt.subplot(122);plt.imshow(img1, clim=clim)
        plt.axis(axes);plt.title('denoised image');plt.colorbar()
        plt.suptitle(title)
        plt.show()
    return img_denoise.detach().cpu().numpy().squeeze()


def apply_RRDB_to_4D_tensor(image):
    # image.shape=(2, 100, 200, 200) -> 2 reference, 100 slices
    s = image.shape
    img_f = torch.zeros_like(image)
    for i in range(s[0]):
        for j in range(s[1]):
            img = image[i:i+1, j:j+1]
            img_f[i, j] = img.squeeze()
    return img_f

def apply_model(model_rrdb, img, device=None):
    """
    Apply a trained RRDB model to a 2D image or 3D image stack.

    Parameters
    ----------
    model_rrdb : torch.nn.Module
        Trained RRDB model.
    img : np.ndarray
        Input image.
        - 2D: (H, W)
        - 3D: (N, H, W), where N is the number of slices.
    device : str or torch.device, optional
        Device to run inference on. If None, automatically uses the
        model's device.

    Returns
    -------
    np.ndarray
        Output image with the same shape as the input.
    """

    if device is None:
        device = next(model_rrdb.parameters()).device

    model_rrdb.eval()

    img = np.asarray(img)

    with torch.no_grad():

        if img.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

            y = model_rrdb(x)

            # (1,1,H,W) -> (H,W)
            return y.squeeze().cpu().numpy()

        elif img.ndim == 3:
            # (N,H,W) -> (N,1,H,W)
            x = torch.from_numpy(img).float().unsqueeze(1).to(device)

            y = model_rrdb(x)

            # (N,1,H,W) -> (N,H,W)
            return y.squeeze(1).cpu().numpy()

        else:
            raise ValueError(
                f"Expected a 2D or 3D array, got shape {img.shape}."
            )

def missing_angle_correct(img_stack, model_rrdb=None, checkpoint_path=None, resize_shape=False, device='cuda'):
    if model_rrdb is None:
        if checkpoint_path is None:
            spec = importlib.util.find_spec("FL")
            fn_model_root = spec.submodule_search_locations[0]  # e.g. '/data/FL_correction/FL'
            fn_model_checkpoint = 'ml_lib/saved_model/multi_scale_rrdb_missing_angle.pth'
            checkpoint_path = os.path.join(fn_model_root, fn_model_checkpoint)
        model_rrdb = load_model_and_checkpoint(checkpoint_path, device)

    s = img_stack.shape
    if len(s) == 2:
        img_stack = img_stack[np.newaxis]
    s = img_stack.shape
    if resize_shape: # model is trained for image size of (200, 200)
        img_stack = resize(img_stack, (s[0], 200, 200))
    img_cor = apply_model(model_rrdb, img_stack, device=None)
    if resize_shape:
        img_cor = resize(img_cor, output_shape=s)

    img_cor = img_cor.squeeze()
    return img_cor, model_rrdb