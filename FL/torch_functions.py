import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from copy import deepcopy
from .cross_section_util import compose_cs_incident_xray_with_reference
from .atten_util import  extract_single_value_param
from .torch_image_util import torch_rotate_image_nd, torch_rot90_4D, torch_rot3D, torch_rot90_4D_tensor
from .cuda_lib.atten_cuda import (
                                cal_atten_cuda,
                                forward_emission_batch,
                                forward_emission,
                                )
from .cuda_lib.autograd import (
                                forward_emission_batch_autograd,
                                forward_emission_autograd,
                                cal_atten_autograd,
                                )
from .image_util import adaptive_threshold
from .file_io import (write_attenuation,
                      write_attenuation_xray,
                      write_attenuation_fl,
                      )

def rm_nan_torch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Placeholder helper function for removing or handling NaNs in a PyTorch tensor.
    Replace this with your actual rm_nan logic.
    """
    return torch.nan_to_num(tensor, nan=0.0)


def torch_pre_treat(*args) -> torch.Tensor:
    """
    Combine a list of 3D data (with same shape) to a 4D torch.float tensor.
    Compatible with PyTorch Autograd.

    Parameters:
    -----------
    a list of 3D tensors, or one 4D/3D tensor

    Returns:
    --------
    4D PyTorch Tensor (dtype=torch.float32)
    """
    n = len(args)

    if n == 0:
        raise ValueError("Input cannot be empty.")

    if n == 1:
        input_data = args[0]

        # Convert to tensor and enforce float dtype without breaking autograd
        if isinstance(input_data, torch.Tensor):
            data = input_data.to(dtype=torch.float)
        else:
            data = torch.tensor(input_data, dtype=torch.float)

        data = torch.squeeze(data)

        if data.ndim == 4:
            data = rm_nan_torch(data)
            return data

        elif data.ndim == 3:
            data = data.unsqueeze(0)
            data = rm_nan_torch(data)
            return data

        else:
            raise ValueError("Input should be either 3D or 4D data.")

    if n > 1:
        # Convert all inputs to torch.Tensor and cast to float
        tensor_list = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_list.append(arg.to(dtype=torch.float))
            else:
                tensor_list.append(torch.tensor(arg, dtype=torch.float))

        # Stack elements along a new dimension
        data = torch.stack(tensor_list)
        data = rm_nan_torch(data)
        return data

def torch_cal_frac(*args, scale_range=[0.02, 0.98], enable_scale=True, scale_limit=0.95):
    """
    len(args) >=1, and all images in *args are torch tensor
    """
    n = len(args)

    device = args[0].device
    if n == 1:
        img_cuda = args[0].clone()
    else:
        img_cuda = torch.stack(args, dim=0)
    n = img_cuda.shape[0]

    img_sum_cuda = torch.sum(img_cuda, axis=0)

    img_np = img_cuda.detach().cpu().numpy()
    img_sum = np.sum(img_np, axis=0) + 1e-12
    img_mask = adaptive_threshold(img_sum, fill_hole=True, dilation=2)
    if np.max(img_mask) == 0: # fails in adaptive_threshold
        img_mask = 1

    img_mask_cuda = torch.tensor(img_mask, dtype=torch.float, device=device)
    img_sum_cuda = img_sum_cuda * img_mask_cuda


    img_sum_sort_cuda = torch.sort(img_sum_cuda[img_sum_cuda > 0])[0]

    ratio_cuda = 1
    l = len(img_sum_sort_cuda)
    pix_max_cuda = torch.median(img_sum_sort_cuda[int(l * scale_limit)])
    if enable_scale:
        try:
            ratio_cuda = img_sum_cuda / pix_max_cuda
        except:
            pix_max_cuda = img_sum_cuda[-1]
            ratio_cuda = 1

    frac_slices = []

    for i in range(n):
        tmp0 = img_cuda[i] / img_sum_cuda
        tmp_cuda = torch.nan_to_num(tmp0, nan=0.0, posinf=0.0, neginf=0.0)
        tmp_cuda = torch.clamp(tmp_cuda, min=0.0)
        if len(scale_range):
            tmp_sort_cuda = torch.sort(tmp_cuda[tmp_cuda>0])[0]
            l = len(tmp_sort_cuda)
            if l > 100:
                tmp_sort_cuda_clamp = tmp_sort_cuda[int(l*scale_range[0]):int(l*scale_range[1])]
                tmp_cuda = torch.clamp(tmp_cuda, min=tmp_sort_cuda_clamp[0], max=tmp_sort_cuda_clamp[-1])
        slice_result = tmp_cuda * img_mask_cuda * ratio_cuda
        frac_slices.append(slice_result)
    frac_cuda = torch.stack(frac_slices, dim=0)
    return frac_cuda




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

    cs_ = deepcopy(param_angle_energy['cs']) # np.array
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
    #n_ele, n_sli, H, W = frac_4D.shape # (n_ele, 100, 100, 100)

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


def torch_xrf_atten_3D(frac_4D_rot,
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
    dtype = frac_4D_rot.dtype
    device = frac_4D_rot.device

    if abs(detector_offset_angle) > 1e-2:
        frac_4D_angle = torch_rotate_image_nd(frac_4D_rot, detector_offset_angle, enable_crop=False)
    else:
        frac_4D_angle = frac_4D_rot

    mu_xrf = torch_extract_3D_mu_xrf(frac_4D_angle, param_angle_energy)

    if position_det == 'r':
        mu_xrf_90 = torch_rot90_4D(mu_xrf, ax=0, mode='clock')
    if position_det == 'l':
        mu_xrf_90 = torch_rot90_4D(mu_xrf, ax=0, mode='c-clock')

    if abs(detector_offset_angle) > 1e-2:
        mu_xrf_90 = torch_rotate_image_nd(mu_xrf_90, detector_offset_angle)  # it is already in cuda_device

    mask_cuda = torch.tensor(detector_mask, dtype=torch.float).to(device)
    mu_xrf_90 = mu_xrf_90.contiguous()
    res = cal_atten_cuda(mu_xrf_90, mask_cuda)
    torch.cuda.synchronize(device)

    # rotate back and reshape the output to the same size of frac_4D
    if abs(detector_offset_angle) > 1e-2:
        res = torch_rotate_image_nd(res, -detector_offset_angle, enable_crop=True)
        s0 = frac_4D_rot.shape
        s1 = frac_4D_angle.shape
        cen_r = s1[2] / 2 - 0.5
        cen_c = s1[3] / 2 - 0.5
        rad_r = s0[2] / 2
        rad_c = s0[3] / 2
        rs = int(max(cen_r - rad_r, 0))
        re = rs + s0[2]
        cs = int(max(cen_c - rad_c, 0))
        ce = cs + s0[3]
        res = res[:, :, rs:re, cs:ce]

    if position_det == 'r':
        res = torch_rot90_4D(res, ax=0, mode='c-clock')
    if position_det == 'l':
        res = torch_rot90_4D(res, ax=0, mode='clock')
    atten_xrf = torch.exp(-res)
    atten_xrf = atten_xrf.cpu().numpy()
    return atten_xrf


def torch_cuda_extract_3D_mu_xray(frac_4D_rot_cuda, param_angle_energy, dict_use_ref_rotate={}):
    """
    Safe out-of-place calculation of mu_xray.
    Uses deepcopy safely since tracking data consists of non-learnable arrays.

    frac_4D: tensor on cuda
    dict_use_ref_rotate['ref_3D']: tensor on cuda
    """
    dtype = frac_4D_rot_cuda.dtype
    device = frac_4D_rot_cuda.device

    cs_ = deepcopy(param_angle_energy['cs'])
    elem_compound = cs_['elem_compound']
    rho = param_angle_energy['rho']
    pix = param_angle_energy['pix']

    if len(dict_use_ref_rotate) > 0:
        use_ref = dict_use_ref_rotate['use_ref']
        ref_3D = dict_use_ref_rotate['ref_3D'] # it is torch.tensor
        ref_comp = dict_use_ref_rotate['ref_comp']

        assert ref_comp is not None, "need to provide 'ref_comp', e.g, 'LiNiO2'"
        #cs_ref = compose_cs_incident_xray_with_reference(ref_3D.detach().cpu().numpy(), cs_, ref_comp)
        cs_ref = torch_cuda_compose_cs_incident_xray_with_reference(ref_3D, cs_, ref_comp)
        cs_[f'{ref_comp}-x'] = cs_ref
        # Collect loop slices out-of-place in a list
    mu_x_slices = []
    for j, elem_comp in enumerate(elem_compound):
        coef = cs_[f'{elem_comp}-x'] * rho[elem_comp] * pix
        if not isinstance(coef, torch.Tensor):
            coef_cuda = torch.tensor(coef, dtype=dtype, device=device)
        else:
            coef_cuda = coef.to(dtype=dtype, device=device)
        mu_x_slices.append(frac_4D_rot_cuda[j] * coef_cuda)
    # Combine everything cleanly without mutating memory
    mu_x_cuda = torch.stack(mu_x_slices, dim=0)

    return mu_x_cuda


def torch_cuda_compose_cs_incident_xray_with_reference(ref_3D, ref_cs, ref_comp):
    n_ref = ref_cs['n_ref']
    assert len(ref_3D) == n_ref, "number of references and number of 3D stack in 'ref_3D' does not match"
    ref_spec = np.zeros(n_ref)
    for i_ref in range(n_ref):
        ref_spec[i_ref] = ref_cs[f'{ref_comp}-x_ref{i_ref + 1}']
    ref_frac_cuda = torch_cal_frac(ref_3D, scale_range=[0.01, 0.99], enable_scale=False, scale_limit=0.99)
    cs_3D = torch.zeros_like(ref_frac_cuda[0])
    for i_ref in range(n_ref):
        cs_3D = cs_3D + ref_spec[i_ref] * ref_frac_cuda[i_ref]
    return cs_3D


def torch_cuda_xray_atten_3D(frac_4D_rotate_cuda, param_angle_energy, dict_use_ref_rotate={}):
    """
    Safe out-of-place calculation of x-ray attenuation without detachment.
    frac_4D_cuda: tensor on cuda
    dict_use_ref_rotate['ref_3D']: tensor on cuda
    """
    mu_x_cuda = torch_cuda_extract_3D_mu_xray(frac_4D_rotate_cuda, param_angle_energy, dict_use_ref_rotate)

    f = 1 - mu_x_cuda
    rev_f = torch.flip(f, dims=(2,))
    rev_cumprod = torch.cumprod(rev_f, dim=2)
    proc = rev_cumprod[:, :, :-1]
    proc = proc.flip(dims=(2,))

    # ✅ FIX: Match the structural dimensions for concatenation.
    # If proc is (n_ele, n_sli, 99, 100), the padding must be (n_ele, n_sli, 1, 100)
    last_slice_ones = torch.ones(
        (proc.shape[0], proc.shape[1], 1, proc.shape[3]),
        dtype=frac_4D_rotate_cuda.dtype,
        device=frac_4D_rotate_cuda.device
    )
    # Concatenate along the flipped dimension (dim=2)
    atten_xray_cuda = torch.cat([proc, last_slice_ones], dim=2)

    return atten_xray_cuda


def torch_cuda_extract_3D_mu_xrf(frac_4D_rotate_cuda, param_angle_energy):
    """
    Safe out-of-place calculation of mu_xrf.
    Preserves the gradient map perfectly across nested element/compound loops.

    frac_4D: torch.tensor (n_elem, n_sli, H, W)
    return:  torch.tensor (n_elem, n_sli, H, W)
    """
    dtype = frac_4D_rotate_cuda.dtype
    device = frac_4D_rotate_cuda.device

    # Safe to keep deepcopy since param_angle_energy contains static NumPy data
    cs_ = deepcopy(param_angle_energy['cs'])
    elem_type = cs_['elem_type']  # e.g., ['Ni', 'Co', 'Mn']
    elem_compound = cs_['elem_compound']  # e.g., ['LiNiO2', 'LiCoO2', 'LiMnO2']
    rho = param_angle_energy['rho']
    pix = param_angle_energy['pix']

    # ✅ FIX 1: Use an outer list to collect each 'xrf_eng' channel out-of-place
    mu_xrf_channels = []

    for i, xrf_eng in enumerate(elem_type):
        # Initialize an accumulator variable for the current fluorescence energy channel
        current_energy_sum = None

        for j, elem_comp in enumerate(elem_compound):
            coef = cs_[f'{elem_comp}-{xrf_eng}'] * rho[elem_comp] * pix

            # ✅ FIX 2: Create the tensor directly on the target device
            if not isinstance(coef, torch.Tensor):
                coef_cuda = torch.tensor(coef, dtype=dtype, device=device)
            else:
                coef_cuda = coef.to(dtype=dtype, device=device)

            # Calculate the contribution for this compound
            contribution = frac_4D_rotate_cuda[j] * coef_cuda

            # ✅ FIX 3: Accumulate out-of-place via standard assignment ('= +')
            if current_energy_sum is None:
                current_energy_sum = contribution
            else:
                current_energy_sum = current_energy_sum + contribution

        # Append the fully accumulated 3D volume for the current element channel
        mu_xrf_channels.append(current_energy_sum)

    # ✅ FIX 4: Stack everything together into a brand-new tensor along dim=0
    mu_xrf_cuda = torch.stack(mu_xrf_channels, dim=0)

    return mu_xrf_cuda


def torch_cuda_xrf_atten_3D(frac_4D_rot_cuda,
                            param_angle_energy,
                            detector_mask,
                            detector_offset_angle=0,
                            position_det='r'
                            ):
    """
    Safely calculates the XRF attenuation out-of-place as a pure PyTorch Tensor.
    Preserves the computational graph completely.

    Input:
    frac_4D_cuda: torch.tensor, (n_ele, sli, H, W)
    detector_mask: 3D array or torch.Tensor
    detector_offset_angle: angle in DEGREES
    """
    dtype = frac_4D_rot_cuda.dtype
    device = frac_4D_rot_cuda.device

    # Handle conditional rotation using torch_rot3D (expects DEGREES)
    if abs(detector_offset_angle) > 1e-2:
        frac_4D_angle_cuda = torch_rot3D(frac_4D_rot_cuda, detector_offset_angle)
    else:
        frac_4D_angle_cuda = frac_4D_rot_cuda

    # Use the safe tensor version of mu_xrf extraction we built
    mu_xrf_cuda = torch_cuda_extract_3D_mu_xrf(frac_4D_angle_cuda, param_angle_energy)

    # Clean out-of-place 90 degree rotation
    if position_det == 'r':
        mu_xrf_90_cuda = torch_rot90_4D_tensor(mu_xrf_cuda, ax=0, mode='clock')
    if position_det == 'l':
        mu_xrf_90_cuda = torch_rot90_4D_tensor(mu_xrf_cuda, ax=0, mode='c-clock')

    if abs(detector_offset_angle) > 1e-2:
        mu_xrf_90_cuda = torch_rot3D(mu_xrf_90_cuda, detector_offset_angle)

    # Safely convert the mask to a tensor if it is a NumPy array
    if not isinstance(detector_mask, torch.Tensor):
        mask_cuda = torch.tensor(detector_mask, dtype=dtype, device=device)
    else:
        mask_cuda = detector_mask.to(dtype=dtype, device=device)

    # Call your safe custom attenuation model
    # res = cal_atten_cuda(mu_xrf_90.to(dtype=dtype), mask_cuda) # without autograd

    mu_xrf_90_cuda = mu_xrf_90_cuda.contiguous()
    res_cuda = cal_atten_autograd(mu_xrf_90_cuda.to(dtype=dtype), mask_cuda) # with autograd
    #res = torch_cuda_cal_atten_differentiable(mu_xrf_90_cuda.to(dtype=dtype), mask_cuda) # with autograd but slower
    torch.cuda.synchronize(device)

    # Rotate back and reshape the output to the same size of frac_4D
    if abs(detector_offset_angle) > 1e-2:
        res = torch_rot3D(res_cuda, -detector_offset_angle)

        # Safe slicing logic using bounding indexes
        s0 = frac_4D_rot_cuda.shape
        s1 = frac_4D_angle_cuda.shape
        cen_r = s1[2] / 2.0 - 0.5
        cen_c = s1[3] / 2.0 - 0.5
        rad_r = s0[2] / 2.0
        rad_c = s0[3] / 2.0

        rs = int(max(cen_r - rad_r, 0))
        re = rs + s0[2]
        cs = int(max(cen_c - rad_c, 0))
        ce = cs + s0[3]
        res_cuda = res_cuda[:, :, rs:re, cs:ce]

    # Clean out-of-place 90 degree rotation back
    if position_det == 'r':
        res_cuda = torch_rot90_4D_tensor(res_cuda, ax=0, mode='c-clock')
    elif position_det == 'l':
        res_cuda = torch_rot90_4D_tensor(res_cuda, ax=0, mode='clock')
    # ✅ FIX 2: Calculate attenuation exponentially out-of-place and return the live tensor
    atten_xrf_cuda = torch.exp(-res_cuda)
    return atten_xrf_cuda


def torch_cuda_cal_atten_differentiable(mu, mask):
    """
    Differentiable pure-PyTorch replacement for cal_atten_cuda.
    Replicates the exact pixel accumulation behavior of atten_kernel
    using 3D cross-correlation/convolution.

    It is "equivalent" to cal_atten_autograd, but slower.

    mu:   Shape (K, Z, X, Y) -> (Batch/Elements, Depth/Slices, Height, Width)
    mask: Shape (Mz, Mx, My) -> Custom 3D filter kernel
    """
    # 1. format mu for F.conv3d -> expects (Minibatch, In_Channels, D, H, W)
    # We treat K as the batch dimension, and add a single channel dimension (1)
    mu_input = mu.unsqueeze(1)  # Shape: (K, 1, Z, X, Y)

    # 2. format mask for F.conv3d -> expects (Out_Channels, In_Channels/Groups, dD, dH, dW)
    # Invert/flip the spatial layout if matching exact index offsets,
    # but since it's an accumulation neighborhood, we wrap it as an explicit filter weight:
    weight = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, Mz, Mx, My)

    # 3. Calculate padding sizes dynamically to match the C++ bounds:
    # C++ loops: z = sli + dz - Mz/2;  x = row + dx;  y = col + dy - My/2
    # This translates to specific asymmetric or center-padded convolutions.
    pad_z = mask.shape[0] // 2
    pad_x = mask.shape[1] - 1  # C++ uses 'x = row + dx' without a backward offset for row
    pad_y = mask.shape[2] // 2

    # Pad the tensor out-of-place safely
    mu_padded = F.pad(mu_input, (pad_y, pad_y, 0, pad_x, pad_z, pad_z), mode='constant', value=0.0)

    # 4. Run out-of-place conv3d (PyTorch automatically tracks gradients through this perfectly)
    res = F.conv3d(mu_padded, weight, stride=1)

    # 5. Restore original layout shape (K, Z, X, Y)
    return res.squeeze(1)


##########################################################################
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
        prj = forward_emission_batch(atten_cuda.contiguous(), C.contiguous(), em_cs_cuda, theta_cuda)
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





def torch_cuda_update_dict_use_ref_at_angle(dict_use_ref_cuda, current_angle, device='cuda'):
    """
    Optimized loop-free dictionary update that rotates a single
    multi-dimensional ref_3D tensor safely out-of-place.

    dict_use_ref_cuda['ref_3D']: tensor on cuda
    """
    # Create a shallow copy of the dictionary to avoid mutating the original
    dict_use_ref_angle = dict_use_ref_cuda.copy()

    if len(dict_use_ref_angle) > 0:
        if dict_use_ref_angle.get('use_ref', False):
            ref_3D_cuda = dict_use_ref_angle['ref_3D']
            # This directly overwrites the dictionary key with the new tensor pointer
            dict_use_ref_angle['ref_3D'] = torch_rot3D(ref_3D_cuda, current_angle)
    return dict_use_ref_angle


def torch_cuda_cal_xray_atten_at_angles(frac_4D_cuda,
                                        param,
                                        angle_list,
                                        dict_use_ref_cuda={},
                                       ):
    """
    frac_4D_cuda:  tensor on cuda
    dict_use_ref_cuda['ref_3D']: tensor on cuda
    """
    n_angle = len(angle_list)
    device = frac_4D_cuda.device
    atten_slices = []
    #angle_cuda = torch.tensor(angle_list, device=device)
    for ang_id in trange(n_angle):
        dict_use_ref_rotate = torch_cuda_update_dict_use_ref_at_angle(dict_use_ref_cuda, angle_list[ang_id])
        param_angle_energy = extract_single_value_param(param,
                                                        ang_id,
                                                        dict_use_ref_rotate
                                                        )
        frac_4D_rot_cuda = torch_rot3D(frac_4D_cuda, angle_list[ang_id])
        atten_xray_cuda = torch_cuda_xray_atten_3D(frac_4D_rot_cuda,
                                            param_angle_energy,
                                            dict_use_ref_rotate,
                                            )

        atten_slices.append(atten_xray_cuda)
    atten_xray_4D_cuda = torch.stack(atten_slices, dim=0)
    return atten_xray_4D_cuda


def torch_cuda_cal_xrf_atten_at_angles(frac_4D_cuda,
                                        param,
                                        angle_list,
                                        detector_mask,
                                        detector_offset_angle=0,
                                        position_det='r',
                                        dict_use_ref_cuda={},
                                        ):
    """
    frac_4D should be a tensor
    dict_use_ref: if not empty, then dict_use_ref['ref_3D'] should be a tensor
    """
    n_angle = len(angle_list)
    elem_type = param['elem_type']  # ['Ni', 'Co', 'Mn']
    atten_slices = []
    for ang_id in trange(n_angle):
        dict_use_ref_rotate_cuda = torch_cuda_update_dict_use_ref_at_angle(dict_use_ref_cuda,
                                                                           angle_list[ang_id])
        param_angle_energy = extract_single_value_param(param,
                                                        ang_id,
                                                        dict_use_ref_rotate_cuda
                                                        )
        frac_4D_rot_cuda = torch_rot3D(frac_4D_cuda, angle_list[ang_id])
        torch.cuda.synchronize()
        atten_xrf_cuda = torch_cuda_xrf_atten_3D(frac_4D_rot_cuda,
                                                param_angle_energy,
                                                detector_mask,
                                                detector_offset_angle,
                                                position_det,
                                                )
        atten_slices.append(atten_xrf_cuda)
    atten_xrf_4D_cuda = torch.stack(atten_slices, dim=0)
    return atten_xrf_4D_cuda


def torch_cuda_save_atten_4D(att, elem_type, fsave, mode):
    """
    att: atten_xray_4D_cuda, atten_xrf_4D_cuda, atten_total_4D_cuda
    mode:
        incident xray: 'xray',
        xray fluorescence: 'xrf' or 'fl',
        total attenuation: 'total' or 'all'

    """
    att_np = att.detach().cpu().numpy()
    s = att_np.shape # e.g, (180, 3, 50, 160, 150) --> (n_angle, n_elem, n_sli, n_row, n_col)

    n_angle = s[0]
    if mode == 'total' or mode == 'all':
        write_func = write_attenuation
        msg = 'save total attenuation'
    elif mode == 'xray':
        write_func = write_attenuation_xray
        msg = 'save xray attenuation'
    elif mode == 'fl' or mode == 'xrf':
        write_func = write_attenuation_fl
        msg = 'save xrf attenuation'
    else:
        print('unknown mode, mode options: "total", "xray", "fl" or "xrf"')
        return None
    print(msg)
    for id_elem, elem in enumerate(elem_type):
        print(f'{elem}')
        for i in trange(n_angle):
            write_func(elem, att_np[i, id_elem], i, fsave)




















