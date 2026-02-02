import numpy as np
import torch

def FL_forward_simple(guess_cuda, angle_list, rho_compound_2D, em_cs_elem, pix, atten2D_at_angle=1, device='cpu'):
    theta_list = angle_list / 180. * np.pi
    n_angle = len(theta_list)
    rho_comp = torch.tensor(rho_compound_2D, dtype=torch.float32).to(device)
    em = torch.tensor(em_cs_elem, dtype=torch.float32).to(device)
    try:
        atten2D = convert_to_cuda_general(atten2D_at_angle, device)
    except:
        atten2D = torch.ones(n_angle, dtype=torch.float32).to(device)

    s = guess_cuda.shape
    xrf_sino = torch.ones((n_angle, 1, 1, s[-1])).to(device)  # (60, 1, 1, 128)
    for i in range(n_angle):
        t = guess_cuda * rho_comp * pix
        t = rot_img_general(t, theta_list[i], device)  # (1, 1, 128, 128)
        t = t * atten2D[i:i + 1]
        xrf_sino[i] = torch.sum(t, axis=-2)  * em[i] # (1, 1, 128)
    return xrf_sino


def tomo_recon_2D(guess_ini, sino_sli, angle_list, em_cs_elem, rho_compound_2D, pix, atten2D_at_angle, loss_r, lr=0.1, n_epoch=50, device='cuda'):
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value': [], 'rate': loss_r[k]}
    loss_his['img_diff'] = {'value': [], 'rate': 1}
    guess_ini = guess_ini[:, np.newaxis]
    guess = torch.tensor(guess_ini, dtype=torch.float32).to(device).requires_grad_(True)

    model_obj_lr = [{"params": guess, "lr": lr}]
    lr_para = {"betas": (0.9, 0.999), "amsgrad": False}
    optimizer = torch.optim.Adam(model_obj_lr, **lr_para)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, threshold=1e-16,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-12)

    sino_sum_gt_cuda = convert_to_cuda_general(sino_sli, device)