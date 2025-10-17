import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

#from examples.HXN_sparse_tomo.data_process import sino_Ni_sli


def tv_loss_norm(c):
    n = torch.numel(c)
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    loss = loss / n
    return loss


def tv_loss_2D(c):
    n = torch.numel(c)
    x = c[1:,:] - c[:-1,:]
    y = c[:,1:] - c[:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    loss = loss / n
    return loss


def l1_loss(inputs, targets):
    loss = nn.L1Loss()
    output = loss(inputs, targets)
    return output


def poisson_likelihood_loss(inputs, targets):
    # loss = inputs - targets + targets * log(targets/inputs)
    n = torch.numel(inputs)
    id_inputs = inputs > 0
    id_targets = targets > 0
    idx = id_inputs * id_targets
    loss = inputs[idx] - targets[idx] + targets[idx] * torch.log(targets[idx] / inputs[idx])
    loss = torch.sum(loss) / n
    return loss


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rot_img(img, theta, device='cpu', dtype=torch.float32):
    '''
    img.shape = (1, 1, 128, 128) or (n_ref, 1, 128, 128)
    '''
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    img = img.to(dtype)
    img = img.to(device)
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(img.shape[0],1,1).to(device)
    grid = F.affine_grid(rot_mat, img.size(), align_corners=False).type(dtype).to(device)
    img_r = F.grid_sample(img, grid, align_corners=False)
    return img_r

def rot_img_general(img, theta, device='cpu', dtype=torch.float32):
    '''
    img.shape = (4, 1, 128, 128)
    '''
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(img.shape[0],1,1).to(device)
    grid = F.affine_grid(rot_mat, img.size(), align_corners=False).type(dtype).to(device)
    img_r = F.grid_sample(img, grid, align_corners=False)
    return img_r

def torch_xrf_sino2D_with_reference(img4d_cuda, angle_list, em_cs_ref, rho_sli,
                              pix, atten2D_at_angle=[1], device='cuda'):
    """
    generate xrf emission sinogram of single slice, with consideration of x-ray and xrf absorption attenuation

    img4d_cuda: stack of 2D image corresponding to concentration of e.g., Ni2 and Ni3
                shape = (n_ref, 1, 128, 128)
    em_cs_ref: contains emission coefficient for e.g., Ni2 and Ni3
               shape = (n_energy, n_ref), note: num of energy = num of angle
    angle_list: unit of degrees

    rho_sli: mass density of material [g/cm3]
        shape = (128, 128)

    pix: pixel size [cm]

    atten2D_at_angle: attenuation coefficient at all angles
            shape = (n_angle, 128, 128)
    return:
    xrf_sino: sinogram contains n_ref componets
            shape = (n_angle, n_ref, 1, 128)
    """

    s = img4d_cuda.shape # (2, 1, 128, 128)

    n_ref = s[0]
    n_angle = len(angle_list)
    theta_list = angle_list / 180.0 * np.pi
    rho_comp = torch.tensor(rho_sli, dtype=torch.float32).to(device) # (128, 128)
    rho_comp = rho_comp.unsqueeze(0) # (1, 128, 128)
    rho_comp = rho_comp.unsqueeze(0) # (1, 1, 128, 128)

    em = torch.tensor(em_cs_ref)
    if em.shape != (n_angle, n_ref):
        em = torch.ones((n_angle, s[0])).to(device) # (n_angle, 2) = (n_energy, 2)

    atten2D_cuda = torch.tensor(atten2D_at_angle, dtype=torch.float32).to(device)
    if not len(atten2D_cuda) == n_angle:
        atten2D_cuda = torch.ones(n_angle, 1, s[-2], s[-1])

    if len(atten2D_cuda.shape) == 3: # e.g., (60, 128, 128)
        atten2D_cuda = atten2D_cuda.unsqueeze(1).to(device) # (60, 1, 128, 128)

    xrf_sino = torch.ones((n_angle, n_ref, 1, s[-1])).to(device) # (60, 2, 1, 128)
    for i in range(n_angle):
        t = img4d_cuda * rho_comp * pix
        t = rot_img_general(t, theta_list[i], device) # (2, 1, 128, 128)
        t = t * atten2D_cuda[i:i+1]
        t_sum = torch.sum(t, axis=-2) # (2, 1, 128)
        for j in range(n_ref):
            xrf_sino[i, j] = t_sum[j] * em[i, j]
    xrf_sino_sum = torch.sum(xrf_sino, axis=1, keepdims=True) # (60, 1, 1, 128)
    return xrf_sino_sum


def ml_recon_xanes2D_with_reference(guess, sino_sum_gt, angle_list, em_cs_ref, rho_compound_2D,
                              pix, atten2D_at_angle, loss_r, lr, n_epoch, device='cuda'):

    sino_sum_gt_cuda = sino_sum_gt # (60, 1, 128)
    if not torch.is_tensor(sino_sum_gt_cuda):
        sino_sum_gt_cuda = convert_to_4D_tensor(sino_sum_gt_cuda, device) # (60, 1, 1, 128)
    guess_old = 0
    s = sino_sum_gt_cuda.shape
    n_ref = em_cs_ref.shape[-1]
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value': [], 'rate': loss_r[k]}
    loss_his['img_diff'] = {'value': [], 'rate': 1}
    if guess is None:
        guess = torch.zeros((n_ref, 1, s[-1], s[-1]), dtype=torch.float32)
        guess = guess.to(device).requires_grad_(True)  # (2, 1, 128, 128)
    else:
        if not torch.is_tensor(guess):
            guess = torch.tensor(guess, dtype=torch.float32)
        guess = guess.reshape(n_ref, 1, s[-1], s[-1])
        guess = guess.to(device).requires_grad_(True)

    for epoch in trange(n_epoch):
        sino_out = torch_xrf_sino2D_with_reference(guess, angle_list, em_cs_ref, rho_compound_2D, pix,
                                                   atten2D_at_angle, device)
        sino_dif = sino_out - sino_sum_gt_cuda

        loss_val['mse'] = torch.square(sino_dif).mean()
        loss_val['tv_sino'] = tv_loss_norm(sino_dif)
        loss_val['tv_img'] = tv_loss_norm(guess)
        loss_val['l1_sino'] = l1_loss(sino_sum_gt_cuda, sino_out)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out, sino_sum_gt_cuda)

        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                if k == 'mse_fit':
                    if epoch < 100:
                        continue
                loss = loss + loss_val[k] * loss_r[k]

        loss.backward()

        with torch.no_grad():
            guess -= lr * guess.grad
            if epoch > 0:
                guess[guess < 0] = 0
            guess.grad = None

        guess_new = guess.detach().cpu().numpy().squeeze()
        loss_his['img_diff']['value'].append(np.abs(np.mean(guess_new - guess_old)))
        guess_old = guess_new
    sino_out = sino_out.detach().cpu().numpy()
    # guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess_new, sino_out


def convert_to_4D_tensor(img, device):
    """
    convert image to 4D torch.tensor, and put on device ('cpu' or 'cuda')
    e.g., (128, 128) -->(1, 1, 128, 128)
    e.g., (4, 128, 128) --> (4, 1, 128, 128)
    """
    s = img.shape # e.g. (4, 128, 128): 4 components (references), or (128, 128) for regular image
    if len(s) == 2:
        img_c = img.reshape((1, 1, *s))
    elif len(s) == 3:
        img_c = img.reshape((s[0], 1, *s[1:]))
    else:
        img_c = img
    img_cuda = torch.tensor(img_c, dtype=torch.float32).to(device)
    return img_cuda

def plot_img(img, sli=0, return_flag=False):
    im = img
    if torch.is_tensor(img):
        im = img.detach().cpu().numpy()
    im = np.squeeze(im)
    s = img.shape
    if len(s) == 3:
        im = im[sli]
    plt.figure()
    plt.imshow(im)
    plt.colorbar()
    if return_flag:
        return im


def plot_loss(h_loss):
    keys = list(h_loss.keys())
    n = len(keys)

    n_r = int(np.floor(np.sqrt(n)))
    n_c = int(np.ceil(n / n_r))
    fig, axes = plt.subplots(nrows=n_r, ncols=n_c, figsize=(12,6))
    idx = 0
    for r in range(n_r):
        for c in range(n_c):
            if idx >= n:
                break
            k = keys[idx]
            try:
                rate = np.array(h_loss[k]['rate'])
                rate[rate==0] = 1
                val = np.array(h_loss[k]['value'])
                val_scale = val / rate
                axes[r, c].plot(val_scale, '-', label=k)
                axes[r, c].legend()
            except:
                print(k)
            idx += 1


def init_guess(guess, sino_shape, n_ref, device):
    s = sino_shape

    if guess is None:
        guess = torch.zeros((n_ref, 1, s[-1], s[-1]), dtype=torch.float32)
        guess = guess.requires_grad_(True)  # (2, 1, 128, 128)
    else:
        if not torch.is_tensor(guess):
            guess = torch.tensor(guess, dtype=torch.float32)
        guess = guess.reshape(n_ref, 1, s[-1], s[-1])
        guess = guess.requires_grad_(True)
    guess = guess.to(device)
    return guess


def FL_forward(guess_cuda, angle_list, n_ref, rho_compound_2D, em_cs_ref, pix, atten2D_at_angle, device):
    """
    if n_ref > 1:
        will calculate the projection based on xanes2D given the reference spectrum
        em_cs_ref: (n_energies, n_ref) e.g., (60, 2) --> 2 reference spectrum
    if n_ref = 1:
        regular tomographic forward projection
        em_cs_ref: (n_energies), e.g. (60,) --> 1 spectrum

    """

    theta_list = angle_list / 180. * np.pi
    n_angle = len(theta_list)
    rho_comp = torch.tensor(rho_compound_2D, dtype=torch.float32).to(device)
    em = torch.tensor(em_cs_ref, dtype=torch.float32).to(device)
    try:
        atten2D = convert_to_4D_tensor(atten2D_at_angle, device)
    except:
        atten2D = torch.ones(n_angle, dtype=torch.float32).to(device)

    s = guess_cuda.shape
    xrf_sino = torch.ones((n_angle, n_ref, 1, s[-1])).to(device)  # (60, 2, 1, 128)
    for i_angle in range(n_angle):
        t = guess_cuda * rho_comp
        t = rot_img_general(t, theta_list[i_angle], device)  # (2, 1, 128, 128)
        t = t * atten2D[i_angle:i_angle + 1]
        t_sum = torch.sum(t, axis=-2)  # (2, 1, 128)
        if n_ref > 1:
            for i_ref in range(n_ref):
                xrf_sino[i_angle, i_ref] = t_sum[i_ref] * em[i_angle, i_ref] * pix
        else:
            xrf_sino[i_angle] = t_sum * em[i_angle] * pix
    if n_ref > 1:
        xrf_sino_sum = torch.sum(xrf_sino, axis=1, keepdims=True)  # (60, 1, 1, 128)
    else:
        xrf_sino_sum = xrf_sino
    return xrf_sino_sum


import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
def FL_tomo_recon(guess_ini, sino_sli, angle_list, em_cs_ref, rho_compound_2D, pix, atten2D_at_angle, loss_r, lr=1e-3, n_epoch=50, device='cuda'):
    """
    guess_ini:
        xanes: (2, 128, 128)
        regular tomo: (1, 128, 128)
    sino_sli:
        sino of "ground truth", e.g., experimental data, (n_angle, 1, 128)
    angle_list:
        [degrees]
    em_cs_ref:
        emission cross-section of references,
        for xanes: (n_angles, n_ref) [e.g., (60, 2)]
        regular tomo: (n_angles,) [e.g, (60,)]
    rho_compound_2D:
        mass density of material [g/cm3] (128, 128)
    pix:
        pixel size [cm]
    atten2D_at_angle:
        attenuation of x-ray and xrf
        (n_angles, 128, 128) e.g., (60, 128, 128), or
        (n_angles,), or
        1 (without attenuation)
    loss_r:
        dictionary
    lr: learning rate

    """
    s = em_cs_ref.shape
    if len(s) == 2:
        n_ref = em_cs_ref.shape[-1]
    else:
        n_ref = 1
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value': [], 'rate': loss_r[k]}
    loss_his['img_diff'] = {'value': [], 'rate': 1}

    #guess_ini = guess_ini[:, np.newaxis]
    #guess = torch.tensor(guess_ini, dtype=torch.float32).to(device).requires_grad_(True)
    guess = convert_to_4D_tensor(guess_ini, device).requires_grad_(True)
    model_obj_lr = [{"params": guess, "lr": lr}]
    lr_para = {"betas": (0.9, 0.999), "amsgrad": False}
    optimizer = torch.optim.Adam(model_obj_lr, **lr_para)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, threshold=1e-16,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-12)

    sino_sum_gt_cuda = convert_to_4D_tensor(sino_sli, device)
    guess_old = 0
    for epoch in trange(n_epoch):
        optimizer.zero_grad()
        sino_out = FL_forward(guess, angle_list, n_ref, rho_compound_2D, em_cs_ref, pix, atten2D_at_angle, device)
        sino_dif = sino_out - sino_sum_gt_cuda
        loss_val['mse'] = torch.square(sino_dif).mean()
        loss_val['tv_sino'] = tv_loss_norm(sino_dif)
        loss_val['tv_img'] = tv_loss_norm(guess)
        loss_val['l1_sino'] = l1_loss(sino_sum_gt_cuda, sino_out)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out, sino_sum_gt_cuda)

        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                if k == 'mse_fit':
                    if epoch < 100:
                        continue
                loss = loss + loss_val[k] * loss_r[k]

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        guess_new = guess.detach().cpu().numpy().squeeze()
        loss_his['img_diff']['value'].append(np.abs(np.mean(guess_new - guess_old)))
        guess_old = guess_new
    sino_out = sino_out.detach().cpu().numpy()
    # guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess_new, sino_out

def FL_tomo_xanes_2ref(guess_ini, sino_sli, angle_list, em_cs_ref,
                   rho_compound_2D, pix, atten2D_at_angle, loss_r, img_sum,
                   lr=1e-3, n_epoch=50, device='cuda'):
    """
    guess_ini:
        xanes: (2, 128, 128)
        regular tomo: (1, 128, 128)
    sino_sli:
        sino of "ground truth", e.g., experimental data, (n_angle, 1, 128)
    angle_list:
        [degrees]
    em_cs_ref:
        emission cross-section of references,
        for xanes: (n_angles, n_ref) [e.g., (60, 2)]
        regular tomo: (n_angles,) [e.g, (60,)]
    rho_compound_2D:
        mass density of material [g/cm3] (128, 128)
    img_sum:
        constrains: e.g., guess_ini[0] + guess[1] = img_sum
    pix:
        pixel size [cm]
    atten2D_at_angle:
        attenuation of x-ray and xrf
        (n_angles, 128, 128) e.g., (60, 128, 128), or
        (n_angles,), or
        1 (without attenuation)
    loss_r:
        dictionary
    lr: learning rate

    """
    s = em_cs_ref.shape
    if len(s) == 2:
        n_ref = em_cs_ref.shape[-1]
    else:
        n_ref = 1
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value': [], 'rate': loss_r[k]}
    loss_his['img_diff'] = {'value': [], 'rate': 1}

    #guess_ini = guess_ini[:, np.newaxis]
    #guess = torch.tensor(guess_ini, dtype=torch.float32).to(device).requires_grad_(True)
    guess = convert_to_4D_tensor(guess_ini, device).requires_grad_(True)
    img_sum = convert_to_4D_tensor(img_sum, device)
    model_obj_lr = [{"params": guess, "lr": lr}]
    lr_para = {"betas": (0.9, 0.999), "amsgrad": False}
    optimizer = torch.optim.Adam(model_obj_lr, **lr_para)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, threshold=1e-16,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-12)

    sino_sum_gt_cuda = convert_to_4D_tensor(sino_sli, device)
    guess_old = 0
    for epoch in trange(n_epoch):
        optimizer.zero_grad()
        sino_out = FL_forward(guess, angle_list, n_ref, rho_compound_2D, em_cs_ref, pix, atten2D_at_angle, device)
        sino_dif = sino_out - sino_sum_gt_cuda
        img_sum_out = torch.sum(guess, axis=0, keepdim=True)
        loss_val['mse'] = torch.square(sino_dif).mean()
        loss_val['tv_sino'] = tv_loss_norm(sino_dif)
        loss_val['tv_img'] = tv_loss_norm(guess)
        loss_val['l1_sino'] = l1_loss(sino_sum_gt_cuda, sino_out)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out, sino_sum_gt_cuda)
        loss_val['img_sum'] = torch.square(img_sum_out - img_sum).mean()

        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                if k == 'mse_sum':
                    if epoch < 20:
                        continue
                loss = loss + loss_val[k] * loss_r[k]

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        guess_new = guess.detach().cpu().numpy().squeeze()
        loss_his['img_diff']['value'].append(np.abs(np.mean(guess_new - guess_old)))
        guess_old = guess_new
    sino_out = sino_out.detach().cpu().numpy()
    # guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess_new, sino_out

