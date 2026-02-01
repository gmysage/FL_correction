import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.autograd import Variable
import torch.optim as optim

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
    img = img#.to(dtype)
    img = img#.to(device)
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

def torch_rotate_image(img, theta, mode="bilinear"):
    """
    Rotate image(s) by angle theta (radians).

    Parameters
    ----------
    img : torch.Tensor
        Shape (H, W) or (N, H, W)
    theta : float or torch.Tensor
        Rotation angle in radians
    mode : str
        'bilinear' or 'nearest'

    Returns
    -------
    rotated : torch.Tensor
        Same shape as input
    """

    # -----------------------------
    # Normalize input shape
    # -----------------------------
    if img.dim() == 2:
        img = img.unsqueeze(0)   # (1, H, W)
        squeeze_back = True
    elif img.dim() == 3:
        squeeze_back = False
    else:
        raise ValueError("img must have shape (H,W) or (N,H,W)")

    N, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # -----------------------------
    # Make batch + channel explicit
    # -----------------------------
    img = img.unsqueeze(1)  # (N, 1, H, W)

    # -----------------------------
    # Rotation matrix
    # -----------------------------
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=device, dtype=dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # affine matrix: (N, 2, 3)
    affine = torch.zeros((N, 2, 3), device=device, dtype=dtype)
    affine[:, 0, 0] = cos_t
    affine[:, 0, 1] = -sin_t
    affine[:, 1, 0] = sin_t
    affine[:, 1, 1] = cos_t

    # -----------------------------
    # Grid + sampling
    # -----------------------------
    grid = F.affine_grid(
        affine,
        size=img.shape,
        align_corners=True
    )

    rotated = F.grid_sample(
        img,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True
    )

    # -----------------------------
    # Restore shape
    # -----------------------------
    rotated = rotated.squeeze(1)  # (N, H, W)

    if squeeze_back:
        rotated = rotated.squeeze(0)  # (H, W)

    return rotated

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
    if not torch.is_tensor(img):
        img = torch.tensor(img, dtype=torch.float32)
    else:
        img = img.to(torch.float32)
    s = img.shape # e.g. (4, 128, 128): 4 components (references), or (128, 128) for regular image
    if len(s) == 2:
        img = img.reshape((1, 1, *s))
    elif len(s) == 3:
        img = img.reshape((s[0], 1, *s[1:]))
    img = img.to(device)
    return img

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
                # rate[rate==0] = 1
                val = np.array(h_loss[k]['value'])
                # val_scale = val / rate
                axes[r, c].plot(val, '-', label=f'{k}\nr={rate:.1e}')
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


def FL_forward(guess_cuda, angle_list, n_ref, rho_compound_2D, em_cs_ref, pix,
               atten2D_at_angle, device='cuda'):
    """
    if n_ref > 1:
        will calculate the projection based on xanes2D given the reference spectrum
        em_cs_ref: (n_energies, n_ref) e.g., (60, 2) --> 2 reference spectrum
    if n_ref = 1:
        regular tomographic forward projection
        em_cs_ref: (n_energies), e.g. (60,) --> 1 spectrum
    atten2D_at_angle:
        attenuation of x-ray and xrf
        (n_angles, 128, 128) e.g., (60, 128, 128), or
        (n_angles,), or
        1 (without attenuation)
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
        t1 = rot_img_general(t, theta_list[i_angle], device)  # (2, 1, 128, 128)
        t2 = t1 * atten2D[i_angle:i_angle + 1]
        t_sum = torch.sum(t2, axis=-2)  # (2, 1, 128)
        if n_ref > 1:
            for i_ref in range(n_ref):
                xrf_sino[i_angle, i_ref] = t_sum[i_ref] * em[i_angle, i_ref] * pix
        else:
            xrf_sino[i_angle] = t_sum * em[i_angle] * pix
    if n_ref > 1:
        xrf_sino_sum = torch.sum(xrf_sino, axis=1, keepdims=True)  # (60, 1, 1, 128)
    else:
        xrf_sino_sum = xrf_sino
    del atten2D, xrf_sino, t, t_sum
    torch.cuda.empty_cache()
    return xrf_sino_sum




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
            try:
                loss_his[k]['value'].append(loss_val[k].item())
                if loss_r[k] > 0:
                    if k == 'mse_fit':
                        if epoch < 100:
                            continue
                    loss = loss + loss_val[k] * loss_r[k]
            except:
                pass
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
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for vgg_param in vgg19.parameters():
        vgg_param.requires_grad_(False)
    vgg19.to(device).eval()

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
        loss_val['vgg'] = vgg_loss(img_sum_out, img_sum)
        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                if k == 'mse_sum':
                    if epoch < 20:
                        continue
                loss = loss + loss_val[k] * loss_r[k]
        #loss = loss + loss_val['vgg']
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        guess_new = guess.detach().cpu().numpy().squeeze()
        loss_his['img_diff']['value'].append(np.abs(np.mean(guess_new - guess_old)))
        guess_old = guess_new
    sino_out = sino_out.detach().cpu().numpy()
    # guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess_new, sino_out

###################################################
def FL_tomo_xanes_with_xrf_correction(guess_ini, config_atten, config_x_atten, config_train):
    """
    config_x_atten: dictionary: calculating incident x-ray attenuation

        1. config_x_atten['x_atten_flag']: bool
             True --> will calculate incident_x_ray attenuation on the fly
        2. config_x_atten['x_atten_period']: int
            only use if 'x_atten_flag' = True
            e.g., 10 --> will calculate/update incident_x_ray attenuation every 10 epochs
        3. config_x_atten['x_atten_v_other']: numpy array, shape = (100, 100)
            only use if 'x_atten_flag' = True
            composition concentration from other elements
            e.g., x_atten_v_other = Co[40] + Mn[40], shape=(100, 100)

    config_atten: dictionary: paramaters used in attenuation calculation
        1. config_atten['atten2D']: numpy array
            atteunation coefficient
            if config_x_atten['cal_x_atten'] is True,
                'atten2D' should include XRF attenuation of specific element
                (e.g., Ni) and incident x-ray attenuation from other element (e.g., Co, Mn)

            if config_x_atten['cal_x_atten'] is False, '
                atten2D' includes XRF and incident x-ray attenuation from all elements.

        2. config_atten['rho_compound_2D']: numpy array
            mass density in unit of g/cm3

        3. config_atten['em_cs_ref']: 2D array
            emission cross-section for references, e.g, Ni2+, Ni3+

        4. config_atten['cs_ref']: 2D array
            absorption cross-section for references, e.g, Ni2+, Ni3+

        5. config_atten['pix']: float
            pixel size in unit of cm

        6. config_atten['ref_2D']: numpy array, shape = (n_ref, 100, 100)
            optional
            2D oxidation map used to regulate training, e.g, (Ni2_at_0_degree, Ni3_at_0_degree)

        7. config_atten['ref_2D_angle']: float, default = 0
            2D oxidation map measured at angle, e.g, 0 degree

        8. config_atten['sino']: numpy array, shape = (n_angle, 100)
            measured sinogram

        9. config_atten['angle_list']: list

    config_train: dictionary: parameters used in training
        1. config_train['n_epoch']: int
            number of epochs, e.g, 200

        2. config_train['lr']: float
            learning rate, e.g, 0.1

        3. config_train['loss_r']: dictionary
            e.g.,
            loss_r['mse_sino'] = 1  # MSE for sinogram
            loss_r['tv_sino'] = 0
            loss_r['l1_sino'] = 0
            loss_r['tv_img'] = 0 # total variation for image
            loss_r['mes_ref_2D'] = 0  ralated to config_atten['ref_2D']

        4. config_train['device']: 'cuda', 'cpu'

    """

    lr = config_train['lr']
    loss_r = config_train['loss_r']
    n_epoch = config_train['n_epoch']
    device = config_train['device']

    pix = config_atten['pix']
    rho = config_atten['rho_compound_2D']
    sino_sli = config_atten['sino']
    angle_list = config_atten['angle_list']
    em_cs_ref = config_atten['em_cs_ref']
    cs_ref = config_atten['cs_ref']

    try:
        atten2D = config_atten['atten2D'] # (n_angle, 100, 100)
        atten2D = torch.tensor(atten2D).to(device)  # (n_angle, 100, 100)
    except:
        atten2D = 1

    try:
        ref_2D = config_atten['ref_2D'] # (n_ref, 100, 100)
        ref_2D = torch.tensor(ref_2D).to(device)
    except:
        ref_2D = None

    try:
        ref_2D_angle = config_atten['ref_2D_angle'] # e.g., 0 degree
    except:
        ref_2D_angle = 0

    try:
        x_atten_flag = config_x_atten['x_atten_flag'] # True / False
    except:
        x_atten_flag = False
    try:
        x_atten_period = config_x_atten['x_atten_period']  # e.g, 10, update incident_x_atten every 10 epochs
    except:
        x_atten_period = 10
    try:
        v_other = config_x_atten['x_atten_v_other'] # e.g., (2, 100, 100)--> concentration of Co and Mn
        v_other = torch.tensor(v_other).to(device)  # (100, 100)
    except:
        v_other = 0

    # convert to tensor is needed

    sino_sli = convert_to_4D_tensor(sino_sli, device) # (n_angle, 1, 1, 100,)
    rho = torch.tensor(rho, dtype=torch.float32).to(device)

    n_angle = len(angle_list)
    s = em_cs_ref.shape
    if len(s) == 2:
        n_ref = em_cs_ref.shape[-1]
    else:
        n_ref = 1
    keys = list(loss_r.keys())
    loss_his = {}

    for k in keys:
        loss_his[k] = {'value': [], 'rate': loss_r[k]}
    loss_his['img_diff'] = {'value': [], 'rate': 1}


    guess = convert_to_4D_tensor(guess_ini, device).requires_grad_(True) # (2, 1, 100, 100)
    model_obj_lr = [{"params": guess, "lr": lr}]
    lr_para = {"betas": (0.9, 0.999), "amsgrad": False}
    optimizer = torch.optim.Adam(model_obj_lr, **lr_para)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5,
                                  threshold=1e-16, threshold_mode='rel',
                                  cooldown=0, min_lr=0, eps=1e-12)

    if x_atten_flag: # if True, will udpate the x_ray_atten
        x_ray_atten = FL_cal_x_atten_with_ref_elem(guess.detach(), angle_list, v_other, cs_ref, rho, pix, device)
        x_ray_atten_freeze = x_ray_atten.clone()
    else:
        x_ray_atten = 1
    img_log = {}
    for i_ref in range(n_ref):
        img_log[i_ref] = []

    for epoch in trange(n_epoch):
        optimizer.zero_grad()
        if x_atten_flag: # if True, will udpate the x_ray_atten
            if epoch % x_atten_period == 0:
                x_ray_atten = FL_cal_x_atten_with_ref_elem(guess, angle_list, v_other, cs_ref, rho, pix, device)
                x_ray_atten_freeze = x_ray_atten.detach()
            else:
                x_ray_atten = x_ray_atten_freeze
        atten = atten2D * x_ray_atten

        sino_out = FL_forward(guess, angle_list, n_ref, rho, em_cs_ref, pix, atten, device)
        sino_dif = sino_out - sino_sli

        # matching the 2D XANES results
        mse_ref_2D = None
        if ref_2D is not None:
            mse_ref_2D = 0
            for i_ref in range(n_ref-1):
                for j_ref in range(i_ref, n_ref):
                    gi = rot_img_general(guess[i_ref:i_ref+1], ref_2D_angle, device)
                    gj = rot_img_general(guess[j_ref:j_ref+1], ref_2D_angle, device)
                    ti = torch.sum(gi, axis=-2, keepdims=True)
                    tj = torch.sum(gj, axis=-2, keepdims=True)
                    mse_ref_2D = mse_ref_2D + ti * ref_2D[j_ref] - tj * ref_2D[i_ref]

        loss_val = {}
        loss_val['mse_sino'] = torch.square(sino_dif).mean()
        loss_val['tv_sino'] = tv_loss_norm(sino_dif)
        loss_val['tv_img'] = tv_loss_norm(guess)
        loss_val['l1_sino'] = l1_loss(sino_out, sino_sli)
        loss_val['poisson_sino'] = poisson_likelihood_loss(sino_out, sino_sli)
        if mse_ref_2D is None:
            loss_val['mse_ref_2D'] = 0
        else:
            loss_val['mse_ref_2D'] = torch.square(mse_ref_2D).mean()
            #loss_val['tv_img'] = tv_loss_norm(torch.abs(mse_ref_2D))

        loss = 0.0
        for k in keys:
            try:
                loss_his[k]['value'].append(loss_val[k].item())
                if loss_r[k] > 0:
                    loss = loss + loss_val[k] * loss_r[k]
            except:
                pass
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.no_grad():
            guess[:] = torch.clamp(guess, min=0)

        for i_ref in range(n_ref):
            img_log[i_ref].append(guess[i_ref].detach().cpu().numpy().squeeze())
    for i_ref in range(n_ref):
        img_log[i_ref] = np.array(img_log[i_ref])

    res = {}
    res['img'] = guess.detach().cpu().numpy().squeeze()
    res['loss_his'] = loss_his
    res['img_log'] = img_log
    res['sino'] = sino_out.detach().cpu().numpy().squeeze()
    res['atten_x_ray'] = x_ray_atten_freeze.cpu().numpy().squeeze()
    res['atten'] = atten.detach().cpu().numpy().squeeze()
    return res

####################################################

def get_features_vgg19(image, model_feature, layers=None):
    if layers is None:
        layers = {'2': 'conv1_2',
                  '7': 'conv2_2',
                  '16': 'conv3_4',
                  '25': 'conv4_4'
                 }
    features = {}
    x = image
    for idx, layer in enumerate(model_feature):
        x = layer(x)
        if str(idx) in layers:
            features[layers[str(idx)]] = x
    return features

def vgg_loss(outputs, label, vgg19, model_feature=[], device='cuda'):
    if not torch.is_tensor(outputs):
        out = torch.tensor(outputs)
    else:
        out = outputs.clone().detach()
    if not torch.is_tensor(label):
        lab = torch.tensor(label).detach()
    else:
        lab = label.clone()
    lab_max = torch.max(lab)
    out_max = torch.max(out)
    #out = out / lab_max
    out = out / out_max
    lab = lab / lab_max
    if out.shape[1] == 1:
        out = out.repeat(1,3,1,1)
    if lab.shape[1] == 1:
        lab = lab.repeat(1,3,1,1)
    out = out.to(device)
    lab = lab.to(device)

    feature_out = {}
    feature_lab = {}
    feature_out[0] = 0.5*get_features_vgg19(out, vgg19, {'2': 'conv1_2'})['conv1_2']
    #feature_out2 = 0.5*get_features_vgg19(out, vgg19, {'25': 'conv4_4'})['conv4_4']
    feature_out[1] = 0.5 * get_features_vgg19(out, vgg19, {'15': 'conv4_4'})['conv4_4']
    feature_out[2] = 0.5 * get_features_vgg19(out, vgg19, {'5': 'conv4_4'})['conv4_4']

    feature_lab[0] = 0.5*get_features_vgg19(lab, vgg19, {'2': 'conv1_2'})['conv1_2']
    #feature_lab2 = 0.5*get_features_vgg19(lab, vgg19, {'25': 'conv4_4'})['conv4_4']
    feature_lab[1] = 0.5 * get_features_vgg19(lab, vgg19, {'15': 'conv4_4'})['conv4_4']
    feature_lab[2] = 0.5 * get_features_vgg19(lab, vgg19, {'5': 'conv4_4'})['conv4_4']

    feature_loss = 0
    n_feature = len(feature_out)
    for i in range(n_feature):
        feature_loss = feature_loss + nn.MSELoss()(feature_out[i], feature_lab[i])
    return feature_loss


def FL_x_atten_ref_elem(guess_cuda, angle_list, ref_frac, ref_cs, rho, pix, device):
    '''
    assume x-ray is from bottom of the object

    if guess_cuda = (2, 1, 100, 100), x-ray is shining from (2, 1, x, 100)

    ref_frac: shape = (100, 100), volume fraction of reference element, e.g Ni

    ref_cs: shape = (n_angle, n_ref), e.g., (80, 2), incident x-ray cross-section for Ni


    '''

    ref_frac = convert_to_4D_tensor(ref_frac, device)  # (1, 1, 100, 100)
    # guess_cuda = convert_to_4D_tensor(guess_cuda, device)
    s = guess_cuda.shape  # (2, 1, 100, 100)
    guess_sum = torch.sum(guess_cuda, axis=(0, 1)) + 1e-12  # to avoid deviding by 0
    # guess_max = torch.max(guess_sum)
    # guess_frac = guess_cuda / guess_max
    guess_frac = guess_cuda / guess_sum
    guess_frac = guess_frac.to(device)
    for i in range(s[0]):
        guess_frac[i] = guess_frac[i] * ref_frac[0]  # (2, 1, 100, 100)

    theta_list = angle_list / 180. * np.pi
    n = len(angle_list)
    n_ref = len(guess_cuda)

    x_ray_atten = torch.ones((n, s[-2], s[-1])).to(device)
    for i_angle in range(n):
        cs_angle = 0
        frac_angle = rot_img_general(guess_frac, theta_list[i_angle], device)
        for i_ref in range(n_ref):
            cs_angle = cs_angle + ref_cs[i_angle][i_ref] * frac_angle[i_ref]
        mu_angle = cs_angle * rho  # (1, 100, 100)
        tmp = torch.ones((s[-2], s[-1])).to(device)
        for j in range(s[-2] - 2, -1, -1):
            t = mu_angle[:, j + 1] * pix
            tmp[j] = (1 - t) * tmp[j + 1]
            # print(tmp[j][40])
            # x_ray_atten[i, j] = x_ray_atten[i, j+1] * (1 - t)
        x_ray_atten[i_angle] = tmp
    return x_ray_atten


def FL_cal_x_atten_with_ref_elem(guess_cuda, angle_list, v_other, cs_ref, rho, pix, device):
    '''
    all should be on CUDA device

    v_other: volume sum from all other element, e.g., Co[40] + Mn[40]

    assume x-ray is from bottom of the object
    if guess_cuda = (2, 1, 100, 100), x-ray is shining from (2, 1, x, 100)

    ref_frac: shape = (100, 100), volume fraction of reference element, e.g Ni

    ref_cs: shape = (n_angle, n_ref), e.g., (80, 2), incident x-ray cross-section for Ni

    '''

    s = guess_cuda.shape  # (2, 1, 100, 100)

    guess_sum = torch.sum(guess_cuda, axis=(0, 1)) + v_other
    guess_frac = guess_cuda / torch.max(guess_sum)

    theta_list = torch.tensor(angle_list / 180. * np.pi).to(device)
    n = len(angle_list)
    n_ref = len(guess_cuda)

    x_ray = torch.ones((n, s[-2], s[-1])).to(device)

    type_rho = str(type(rho))
    if 'array' in type_rho or 'Tensor' in type_rho:
        rho = convert_to_4D_tensor(rho, device)

    for i_angle in range(n):
        mu_angle = 0
        frac_angle = rot_img_general(guess_frac*rho, theta_list[i_angle], device)
        for i_ref in range(n_ref):
            mu_angle = mu_angle + cs_ref[i_angle][i_ref] * frac_angle[i_ref]
        #mu_angle = cs_angle * rho  # (1, 100, 100)

        """
        # old inplace code, not working
        for j in range(s[-2]-2, -1, -1):
            t = mu_angle[:, j+1] * pix
            x_ray[i_angle, j] = x_ray[i_angle, j+1] * (1 - t)
        cs_angle = cs_angle.detach()
        """

        # --- vectorized replacement for the backward column-wise loop ---
        # mu_angle: shape (1, H, W)
        # desired x_row[j] = prod_{k=j+1..H-1} (1 - mu_angle[:,k,:]*pix), and x_row[H-1] = 1

        f = 1.0 - (mu_angle * pix)  # shape (1, H, W)
        H = f.shape[1]

        # If H == 1, x_row is just ones
        if H == 1:
            x_row = torch.ones((1, 1, f.shape[2]), device=f.device, dtype=f.dtype)
        else:
            # reverse f along the column dim, compute cumulative product from the end
            rev_f = torch.flip(f, dims=[1])  # shape (1, H, W)
            rev_cumprod = torch.cumprod(rev_f, dim=1)  # shape (1, H, W)
            # rev_cumprod[:, 0] == f_{H-1}, rev_cumprod[:, H-1] == prod_{k=H-1..0} f_k

            # we need x_row[j] = prod_{k=j+1..H-1} f_k
            # take first H-1 elements of rev_cumprod and flip back to align:
            proc = rev_cumprod[:, :H - 1, :]  # shape (1, H-1, W)
            proc = torch.flip(proc, dims=[1])  # shape (1, H-1, W)
            ones_col = torch.ones((1, 1, f.shape[2]), device=f.device, dtype=f.dtype)
            x_row = torch.cat([proc, ones_col], dim=1)  # shape (1, H, W)

        # now x_row.squeeze(0) has shape (H, W) like x_ray[i_angle]
        x_ray[i_angle] = x_row.squeeze(0)
        # --- end vectorized section ---
    return x_ray


def torch_cal_xray_atten(
    C,            # (n_ref, H, W)
    C_other,      # (H, W)
    theta_list,   # (n_angle,)
    cs_ref,       # (n_angle, n_ref)
    rho,          # (H, W)
    pix
):
    device = C.device
    dtype = C.dtype

    n_ref, H, W = C.shape
    n_angle = theta_list.numel()

    C_sum = C.sum(dim=0) + C_other
    C_frac = C / (C_sum.max() + 1e-9)

    atten_xray = torch.empty((n_angle, H, W), device=device, dtype=dtype)

    for i in range(n_angle):
        frac_angle = torch_rotate_image(C_frac, theta_list[i])  # must be grid_sample-based

        mu = torch.zeros((H, W), device=device, dtype=dtype)
        for j in range(n_ref):
            mu += cs_ref[i, j] * frac_angle[j] * rho * pix

        f = torch.clamp(1.0 - mu, min=1e-6)

        rev_f = torch.flip(f, dims=[0])
        rev_prod = torch.cumprod(rev_f, dim=0)

        body = torch.flip(rev_prod[:-1], dims=[0])
        ones = torch.ones((1, W), device=device, dtype=dtype)

        atten_xray[i] = torch.cat([body, ones], dim=0)

    return atten_xray



def torch_cal_xray_atten_vectorize(
    C,          # (n_ref, H, W)
    C_other,    # (H, W)
    theta,      # (n_angle,)
    cs_ref,     # (n_angle, n_ref)
    rho,        # (H, W)
    pix         # scalar
):
    """
    Fully vectorized, differentiable X-ray attenuation
    Returns: atten_xray (n_angle, H, W)
    """

    device = C.device
    dtype = C.dtype

    n_ref, H, W = C.shape
    n_angle = theta.numel()

    # ======================================================
    # Step 1: volume fraction
    # ======================================================
    C_sum = C.sum(dim=0) + C_other            # (H, W)
    C_frac = C / (C_sum.max() + 1e-8)         # (n_ref, H, W)

    # ======================================================
    # Step 2: rotate ALL angles at once
    # ======================================================
    # Expand to (n_angle, n_ref, H, W)
    C_frac = C_frac.unsqueeze(0).expand(n_angle, -1, -1, -1)

    # ---- build affine matrices ----
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    affine = torch.zeros((n_angle, 2, 3), device=device, dtype=dtype)
    affine[:, 0, 0] = cos_t
    affine[:, 0, 1] = -sin_t
    affine[:, 1, 0] = sin_t
    affine[:, 1, 1] = cos_t

    # ---- grid + sample ----
    grid = F.affine_grid(
        affine,
        size=(n_angle, 1, H, W),
        align_corners=True
    )
    # reshape for grid_sample: (N, C, H, W)
    frac_rot = F.grid_sample(
        C_frac.reshape(n_angle * n_ref, 1, H, W),
        grid.repeat_interleave(n_ref, dim=0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )
    frac_rot = frac_rot.reshape(n_angle, n_ref, H, W)
    # ======================================================
    # Step 3: compute μ(x, y, angle)
    # ======================================================
    # cs_ref: (n_angle, n_ref) → (n_angle, n_ref, 1, 1)
    mu = (frac_rot * cs_ref[:, :, None, None]).sum(dim=1) * rho * pix  # (n_angle, H, W)
    # ======================================================
    # Step 4: cumulative attenuation (NO Python loops)
    # ======================================================
    f = torch.clamp(1.0 - mu, min=1e-8)                            # (n_angle, H, W)

    # reverse row direction
    f_rev = torch.flip(f, dims=[1])            # flip H
    # cumulative product
    cumprod = torch.cumprod(f_rev, dim=1)
    # shift & restore
    atten_xray = torch.ones_like(f)
    atten_xray[:, :-1, :] = torch.flip(cumprod[:, 1:, :], dims=[1])

    return atten_xray
