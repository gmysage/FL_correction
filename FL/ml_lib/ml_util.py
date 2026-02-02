import numpy as np
import glob
import torch
import math
import importlib.util
from .ml_model import RefAware3DUNet
from tqdm import tqdm

def load_default_3DUNet_model(id_model=None, device='cuda'):
    spec = importlib.util.find_spec("FL")
    fn_model_root = spec.submodule_search_locations[0] # e.g. '/data/FL_correction/FL'

    if id_model is None:
        #model_path = fn_model_root + '/ml_lib/saved_model/best_3dcnn_0048.pth'
        model_path = fn_model_root + '/ml_lib/saved_model/refaware3dUnet_0263.pth'
    else:
        fn_model = fn_model_root + '/ml_lib/saved_model/saved_model_UNet'
        model_list = np.sort(glob.glob(fn_model + '/*.pth'))
        model_path = model_list[id_model]
    model_3DUNet = RefAware3DUNet(
            n_ref=1,        # single-reference pretraining
            emb_dim=8,
            base_ch=32
            )
    model_3DUNet.load_state_dict(torch.load(model_path, map_location=device))
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