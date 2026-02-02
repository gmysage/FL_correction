import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from .cross_section_util import compose_cs_incident_xray_with_reference
from .torch_image_util import torch_rotate_image_nd, torch_rot90_4D
from .cuda_lib.atten_cuda import cal_atten_cuda, forward_emission_batch, forward_emission
def torch_extract_3D_mu_xray(frac_4D,
                             param_angle_energy,
                             dict_use_ref_rotate={},
                             ):
    """
    frac_4D: torhc.tensor, [n_elem, 100, 100, 100]
    Note: incident x-ray is from the "bottom" to "top" of the 3D volume

    return:
    mu_x: torch.tensor, [n_elem, 100, 100, 100]
    """
    dtype = frac_4D.dtype
    device = frac_4D.device

    if len(dict_use_ref_rotate) > 0:
        use_ref = dict_use_ref_rotate['use_ref']
        ref_3D = dict_use_ref_rotate['ref_3D']
        ref_comp = dict_use_ref_rotate['ref_comp']
    else:
        use_ref = False
    cs_ = deepcopy(param_angle_energy['cs'])
    elem_type = cs_['elem_type']  # 'Ni', 'Co', 'Mn'
    elem_compound = cs_['elem_compound']  # 'LiNiO2', 'LiCoO2', 'LiMnO2'
    rho = param_angle_energy['rho']
    pix = param_angle_energy['pix']

    if use_ref:
        assert ref_comp is not None, "need to provide 'ref_comp', e.g, 'LiNiO2'"
        cs_ref = compose_cs_incident_xray_with_reference(ref_3D, cs_, ref_comp)
        cs_[f'{ref_comp}-x'] = cs_ref  # this is a 3D volume of cross-section

    mu_x = torch.ones_like(frac_4D)
    for j, elem_comp in enumerate(elem_compound):
        # e.g
        # mu['x'] = mu[8.4 keV] = f[LiNiO2] * cs[LiNiO2-8.4keV] * rho[LiNiO2] +
        #                       + f[LiCoO2] * cs[LiCoO2-8.4keV] * rho[LiCoO2] +
        #                       + f[LiMnO2] * cs[LiMnO2-8.4keV] * rho[LiMnO2] +
        # note if use ref:
        # cs[LiNiO2-8.4keV] is a 3D volume
        # cs[LiCoO2-8.4keV] and is cs[LiMnO2-8.4keV] is single value
        coef = cs_[f'{elem_comp}-x'] * rho[elem_comp] * pix
        coef_cuda = torch.tensor(coef, dtype=dtype, device=device)
        mu_x[j] = frac_4D[j] * coef_cuda

    return mu_x


def torch_extract_3D_mu_xrf(frac_4D, param_angle_energy):
    '''
    frac_4D: torch.tensor (n_elem, n_sli, H, W)

    return:
    mu_xrf: torch.tensor, (n_ele, n_sli, H, W)
    '''
    dtype = frac_4D.dtype
    device = frac_4D.device

    cs_ = deepcopy(param_angle_energy['cs'])
    elem_type = cs_['elem_type']  # 'Ni', 'Co', 'Mn'
    elem_compound = cs_['elem_compound']  # 'LiNiO2', 'LiCoO2', 'LiMnO2'
    rho = param_angle_energy['rho']
    pix = param_angle_energy['pix']

    mu_xrf = torch.zeros_like(frac_4D)
    for i, xrf_eng in enumerate(elem_type):  # ['Ni', 'Co', 'Mn']
        for j, elem_comp in enumerate(elem_compound):
            # e.g
            # mu['Ni'] = mu[7.4keV] = f[LiNiO2] * cs[LiNiO2-7.4keV] * rho[LiNiO2] +
            #                       + f[LiCoO2] * cs[LiCoO2-7.4keV] * rho[LiCoO2] +
            #                       + f[LiMnO2] * cs[LiMnO2-7.4keV] * rho[LiMnO2] +
            coef = cs_[f'{elem_comp}-{xrf_eng}'] * rho[elem_comp] * pix
            coef_cuda = torch.tensor(coef, dtype=dtype).to(device)
            mu_xrf[i] += frac_4D[j] * coef_cuda
    return mu_xrf

def torch_xray_atten_3D(frac_4D, param_angle_energy, dict_use_ref_rotate={}, flag_detach=True):
    '''
    Note: incident x-ray is from the "bottom" to "top" of the 3D volume

    frac_4D.shape = (n_ele, 100, 100, 100)
    
    return:
    atten_xray: np.array: (n_ele, 100, 100, 100)
    '''
    mu_x = torch_extract_3D_mu_xray(frac_4D, param_angle_energy, dict_use_ref_rotate)
    n_ele, n_sli, H, W = frac_4D.shape # (n_ele, 100, 100, 100)

    atten_xray = torch.ones_like(frac_4D)

    f = 1 - mu_x # (n_ele, n_sli, H, W)
    rev_f = torch.flip(f, dims=(2,))
    rev_cumprod = torch.cumprod(rev_f, dim=2)
    proc = rev_cumprod[:, :, :-1]
    proc = proc.flip(dims=(2,))
    atten_xray[:, :, :-1]=proc
    if flag_detach:
        atten_xray = atten_xray.detach().cpu().numpy()
    
    return atten_xray



def torch_xrf_atten_3D(frac_4D, 
                    param_angle_energy, 
                    detector_mask,
                    detector_offset_angle=0,
                    position_det='r'
                    ):
    """
    calculate the xrf attenuation

    Input:
    frac_4D: torch.tensor, (n_ele, sli, H, W)

    detector_mask: 3D array

    detector_offset_angle: angle of detector away from 90 degrees to incident xray
            e.g. 0: detector is at 90 deg to incident x-ray
                 15: detortor is at 75 deg to incident x-ray, in this case, need to c-clock rotate data to calculate xrf-atten
    
    return:
    atten_xrf: np.array, (n_ele, sli, H, W)
    """
    dtype = frac_4D.dtype
    device = frac_4D.device
    
    offset_angle = torch.tensor(detector_offset_angle/180. * np.pi, dtype=dtype, device=device)
    if torch.abs(offset_angle) > 1e-2:
        frac_4D_angle = torch_rotate_image_nd(frac_4D, offset_angle, enable_crop=False)
    else:
        frac_4D_angle = frac_4D

    mu_xrf = torch_extract_3D_mu_xrf(frac_4D_angle, param_angle_energy)

    if position_det=='r':
        mu_xrf_90 = torch_rot90_4D(mu_xrf, ax=0, mode='clock')
    if position_det=='l':
        mu_xrf_90 = torch_rot90_4D(mu_xrf, ax=0, mode='c-clock')

    if torch.abs(offset_angle) > 1e-2:
        mu_xrf_90 = torch_rotate_image_nd(mu_xrf_90, offset_angle) # it is already in cuda_device
    mu_xrf_cuda = torch.tensor(mu_xrf_90, dtype=torch.float).to(device)
    mask_cuda = torch.tensor(detector_mask, dtype=torch.float).to(device)
    
    res = cal_atten_cuda(mu_xrf_cuda, mask_cuda)
    torch.cuda.synchronize(device)

    # rotate back and reshape the output to the same size of frac_4D
    if torch.abs(offset_angle) > 1e-2:
        res = torch_rotate_image_nd(res, -offset_angle, enable_crop=True)
        s0 = frac_4D.shape
        s1 = frac_4D_angle.shape
        cen_r = s1[2] / 2 - 0.5
        cen_c = s1[3] / - 0.5
        rad_r = s0[2] / 2
        rad_c = s0[3] / 2
        rs = int(max(cen_r - rad_r, 0))
        re = rs + s0[2]
        cs = int(max(cen_c - rad_c, 0))
        ce = cs + s0[3]
        res = res[:, :, rs:re, cs:ce]

    if position_det=='r':
        res = torch_rot90_4D(res, ax=0, mode='c-clock')
    if position_det=='l':
        res = torch_rot90_4D(res, ax=0, mode='clock')
    atten_xrf = torch.exp(-res)
    atten_xrf = atten_xrf.cpu().numpy()
    return atten_xrf




def torch_2D_xray_atten_with_ref(
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
        frac_angle = torch_rotate_image_nd(C_frac, theta_list[i])  # must be grid_sample-based

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


def torch_3D_forward_projection(C,          # (n_ref, n_sli, H, W) 
                                angle_list, # (n_angle)
                                em_cs,      # (n_angle, n_ref)    
                                atten,       # (n_angle, n_sli, H, W)
                                mode='batch'
                                ):
    dtype = C.dtype
    device = C.device
    theta_cuda = torch.tensor(angle_list/180.*np.pi, dtype=dtype, device=device)   
    em_cs_cuda = torch.tensor(em_cs, dtype=dtype, device=device)
    atten_cuda = torch.tensor(atten, dtype=dtype, device=device)
    if mode == 'batch':
        prj = forward_emission_batch(atten_cuda, C, em_cs_cuda, theta_cuda)
        prj = prj.cpu().numpy()
    else:
        n_angle = len(angle_list)
        n_ref, n_sli, H, W = C.shape
        prj = np.zeros((n_angle, n_sli, W))
        for i in range(n_sli):
            Pf = forward_emission(atten_cuda[:, i].contiguous(),
                                    C[:, i].contiguous(),
                                    em_cs_cuda,
                                    theta_cuda)
            prj[:, i] = Pf.cpu().numpy()
    return prj


def torch_2D_xray_atten_with_ref_vectorize(
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
    mu = torch.relu(mu)
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


