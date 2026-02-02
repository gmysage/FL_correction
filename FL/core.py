import numpy as np
import torch
import glob
import xraylib
import os
from tqdm import trange
from .torch_image_util import torch_rotate_image_nd, torch_rot90_4D
from .torch_functions import torch_xrf_atten_3D, torch_xray_atten_3D
from .atten_util import extract_single_value_param, update_dict_use_ref_at_angle, get_atten_coef
from .file_io import (write_attenuation,
                      write_attenuation_xray,
                      write_attenuation_fl,
                      read_attenuation_at_all_angle
                      )
from .util import pre_treat
from .image_util import adaptive_threshold, rm_abnormal
from .cuda_lib.atten_cuda import forward_emission_batch

def cal_xrf_atten_at_angles(frac_4D,
                            param, 
                            angle_list, 
                            detector_mask,
                            detector_offset_angle=0,
                            position_det='r',
                            dict_use_ref={},
                            fsave='./Angle_prj',
                            ):
    n_angle = len(angle_list)
    dtype = frac_4D.dtype
    device = frac_4D.device

    elem_type = param['elem_type']  # ['Ni', 'Co', 'Mn']
    n_ele = len(elem_type)
    for ang_id in trange(n_angle):
        theta_cuda = torch.tensor(angle_list[ang_id]/180. * np.pi, device=device)
        dict_use_ref_rotate = update_dict_use_ref_at_angle(dict_use_ref,
                                                           angle_list[ang_id]
                                                           )
        param_angle_energy = extract_single_value_param(param,
                                                        ang_id, 
                                                        dict_use_ref_rotate
                                                        )
        frac_4D_rot = torch_rotate_image_nd(frac_4D, theta_cuda)
        torch.cuda.synchronize()
        atten_xrf = torch_xrf_atten_3D(frac_4D_rot, 
                                        param_angle_energy, 
                                        detector_mask,
                                        detector_offset_angle,
                                        position_det,
                                        )
        for i, elem in enumerate(elem_type):
            write_attenuation_fl(elem, atten_xrf[i], ang_id, fsave)


def cal_xray_atten_at_angles(frac_4D, 
                            param, 
                            angle_list, 
                            dict_use_ref={},
                            fsave='./Angle_prj',
                            ):
    n_angle = len(angle_list)
    dtype = frac_4D.dtype
    device = frac_4D.device
    elem_type = param['elem_type']  # ['Ni', 'Co', 'Mn']

    for ang_id in trange(n_angle):
        theta_cuda = torch.tensor(angle_list[ang_id]/180.*np.pi, device=device)
        dict_use_ref_rotate = update_dict_use_ref_at_angle(dict_use_ref,
                                                           angle_list[ang_id]
                                                           )
        param_angle_energy = extract_single_value_param(param,
                                                        ang_id, 
                                                        dict_use_ref_rotate
                                                        )
        frac_4D_rot = torch_rotate_image_nd(frac_4D, theta_cuda)
        atten_xray = torch_xray_atten_3D(frac_4D_rot, 
                                        param_angle_energy, 
                                        dict_use_ref_rotate={}
                                        )
        for i, elem in enumerate(elem_type):
            write_attenuation_xray(elem, atten_xray[i], ang_id, fsave)


def cal_total_atten_at_angles(fsave, param):
    elem_type = param['elem_type']
    n_ele = len(elem_type)
    n_angle = int(len(glob.glob(fsave + '/atten_fl*')) / n_ele)

    for elem in elem_type:
        att_ele_xrf = read_attenuation_at_all_angle(np.arange(n_angle),
                                                    fsave,
                                                    elem,
                                                    'fl')

        att_ele_xray = read_attenuation_at_all_angle(np.arange(n_angle),
                                                     fsave,
                                                     elem,
                                                     'xray')
        att = att_ele_xrf * att_ele_xray
        for i in range(n_angle):
            write_attenuation(elem, att[i], i, fsave)


def re_projection_cuda(img3D, angle_list, param=None, rho_compound=1,
                       atten_coef=None, elem='Ni', use_ref=False, device='cuda'):
    '''
    calculate the xrf emission of the 2D projection from a 3D volume at all angles
    param contains the mass density (rho), pixel size (pix) and emission cross-section (em_cs)

    xrf(pixel) = concentration(pixel) * cs * rho * pixel_size

    if use_ref = True, it will read emission cross-section value from:
        param['em_cs']['Ni_ref1']
        param['em_cs']['Ni_ref2']

    Input:
    img3D: (sli, H, W), or (n_ref, sli, H, W)
    angle_list: (n_angle,)
    atten_coef: (n_angle, sli, H, W)
    '''
    n_angle = len(angle_list)
    if param is None:
        em = np.ones((n_angle, 1))
    else:
        pix = param['pix']
        em_cs_dict = param['em_cs']

        if use_ref:
            n_ref = img3D.shape[0]
            em = np.zeros([n_angle, n_ref])
            for i in range(n_ref):
                em[:, i] = em_cs_dict[f'{elem}_ref{i + 1}'] * pix
        else:
            em = em_cs_dict[elem] * pix
            em = em.reshape((len(em), 1))

    img3D = img3D * rho_compound
    if len(img3D.shape) == 3:
        img3D = img3D[np.newaxis]  # (1, sli, H, W)

    C = torch.tensor(img3D, dtype=torch.float, device=device)

    n_ref, n_sli, H, W = C.shape

    if atten_coef is None:
        atten_cuda = torch.ones(n_angle, n_sli, H, W, dtype=torch.float, device=device)
    else:
        if len(atten_coef.shape) == 3:
            atten_coef = atten_coef[:, np.newaxis]
        atten_cuda = torch.tensor(atten_coef, dtype=torch.float, device=device)
    theta_cuda = torch.tensor(angle_list / 180. * np.pi, dtype=torch.float, device=device)

    em_cs_cuda = torch.tensor(em, dtype=torch.float, device=device)
    prj = forward_emission_batch(atten_cuda, C, em_cs_cuda, theta_cuda)
    prj = prj.cpu().numpy()
    return prj


def cal_frac(*args, scale_range=[0.02, 0.98], enable_scale=True, scale_limit=0.95):
    if len(args) == 1:
        img = args[0].copy()
        s = img.shape
        if len(s) == 2:
            img = np.expand_dims(img, 0)
    else:
        img = pre_treat(*args)

    s = img.shape
    n = s[0]
    frac = np.zeros(s)
    img_sum = np.sum(img, axis=0) + 1e-12
    '''
    sum_sort = np.sort(img_sum[img], axis=None)
    pix_max = sum_sort[int(0.95*len(sum_sort))]
    '''

    img_mask = adaptive_threshold(img_sum, fill_hole=True, dilation=2)
    if np.max(img_mask) == 0: # fails in adaptive_threshold
        img_mask = 1
    img_sum =  img_sum * img_mask
    sum_sort = np.sort(img_sum[img_sum > 0])
    ratio = 1
    l = len(sum_sort)
    pix_max_update = np.median(sum_sort[int(l * scale_limit)])
    if enable_scale:
        try:
            ratio = img_sum / pix_max_update
        except:
            pix_max_update = sum_sort[-1]
            ratio = 1

    pix_median = np.median(sum_sort)

    for i in range(n):
        tmp = rm_abnormal(img[i] / img_sum)
        if len(scale_range):
            tmp_sort = np.sort(tmp[tmp>0])
            l = len(tmp_sort)
            if l > 100:
                tmp_sort = tmp_sort[int(l*scale_range[0]):int(l*scale_range[1])]
                tmp[tmp >= tmp_sort[-1]] = tmp_sort[-1]
                tmp[tmp <= tmp_sort[0]] = tmp_sort[0]
        frac[i] = tmp * img_mask * ratio

    res = {}
    res['img_sum'] = img_sum
    res['frac'] = frac
    res['pix_median'] = pix_median
    res['pix_max'] = pix_max_update
    res['img_mask'] = img_mask
    return res

def load_param(fn):

    """
    Read parameter files with format of:

    X-ray Energy:	  12
    Number of elements:	   3
    Element type: Zr, La, Hf
    Pixel size(nm):    50
    density(g/cm3):    5.11
    emission line energy: 2.044, 4.647, 7.899
    emission cross section (cm2/g): 1.313, 3.421, 14.97

    """

    with open(fn) as f:
        lines = f.readlines()

    lines = [x.strip('\n') for x in lines]
    lines = [x.replace('\t', ' ') for x in lines]
    lines = [x for x in lines if x]
    lines = [x.strip(' ') for x in lines]

    for i in range(np.size(lines)):
        '''
        if lines[i].find('#') >-1:
            continue
        '''
        words = lines[i].split('#')[0]
        words = words.split(':')
        if words[0].lower().find('x-ray energy')>-1:
            tmp = words[1].strip(' ')
            if os.path.exists(tmp):
                XEng_list = np.loadtxt(tmp)
            else:
                XEng_list = np.array([float(words[1].strip(' '))])
            print(f'XEng = {XEng_list} keV')

        elif words[0].lower().find('number')>-1:
            nelem = int(words[1].strip(' ' ))
            print(f'num of elem = {nelem}')

        elif words[0].lower().find('thickness') > -1:
            thick = float(words[1].strip(' '))
            print(f'max thickness = {thick} um')

        elif words[0].lower().find('pixel')>-1:
            pix = float(words[1].strip(' '))
            pix = pix * 1e-7 # unit of cm
            print(f'pix size = {pix:3.1e} cm')

        elif words[0].lower().find('density')>-1:
            #rho = float(words[1].strip(' '))
            rho = [float(x.strip(' ')) for x in words[1].split(',')]
            print(f'mass density = {rho} g/cm3')

        elif words[0].lower().find('emission energy')>-1:
            em_eng = [float(x.strip(' ')) for x in words[1].split(',')]
            print(f'em_eng = {em_eng} keV')

        elif words[0].lower().find('element type')>-1:
            elem_type = [x.strip(' ') for x in words[1].split(',')]
            print(f'element type: {elem_type}')

        elif words[0].lower().find('element compound') > -1:
            elem_comp = [x.strip(' ') for x in words[1].split(',')]
            print(f'element compound: {elem_comp}')

        elif words[0].lower().find('shell')>-1:
            xrf_shell = [x.strip(' ') for x in words[1].split(',')]
            print(f'XRF_shell: {xrf_shell}')

    M = {} # mole mass of compound
    em_E = {} # emission energy (keV)
    em_cs = {} # emission cross section (cm2/g)
    rho_elem = {} # mass density for compond containing each element (g/cm3)
    for i in range(nelem):
        ele = elem_type[i]
        ele_comp = elem_comp[i]
        atom_idx = xraylib.SymbolToAtomicNumber(ele)
        em_E[ele] = em_eng[i]
        rho_elem[ele_comp] = rho[i]
        # M[ele] = xraylib.AtomicWeight(atom_idx)
        M[ele_comp] = xraylib.CompoundParser(ele_comp)['molarMass']

        if xrf_shell[i] == 'K':
            em_cs[ele] = []
            for i, XEng in enumerate(XEng_list):
                try:
                    current_cs = xraylib.CS_FluorLine(atom_idx, xraylib.KA_LINE, XEng)
                    current_cs += xraylib.CS_FluorLine(atom_idx, xraylib.KB_LINE, XEng)
                except:
                    # in case The excitation energy too low to excite the shell,
                    # set to a large number, when normalize the XRF with emission cross-section, it will effectively
                    # reduce the xrf counts
                    current_cs = 0
                em_cs[ele].append(current_cs)
            em_cs[ele] = np.array(em_cs[ele])
            valid_em_cs = em_cs[ele][em_cs[ele]>0]
            idx = em_cs[ele] < 0
            em_cs[ele][idx] = np.max(valid_em_cs)
        elif xrf_shell[i] == 'L':
            em_cs[ele] = []
            for XEng in XEng_list:
                try:
                    current_cs = xraylib.CS_FluorLine(atom_idx, xraylib.LA_LINE, XEng)
                    current_cs += xraylib.CS_FluorLine(atom_idx, xraylib.LB_LINE, XEng)
                except: # in case The excitation energy too low to excite the shell
                    current_cs = 1e-2
                em_cs[ele].append(current_cs)
    em_cs['x'] = XEng_list

    # get absorption cross-section
    cs = get_atten_coef(elem_type, elem_comp, XEng_list, em_E)

    res = {}
    res['XEng'] = XEng_list
    res['nelem'] = nelem
    res['rho'] = rho_elem
    res['pix'] = float(f'{pix:3.1e}')
    res['M'] = M
    res['em_E'] = em_E
    res['em_cs'] = em_cs
    res['elem_type'] = elem_type
    res['elem_compound'] = elem_comp
    res['eng_list'] = XEng_list
    res['cs'] = cs
    print(f'\nkeys in param:\n{list(res.keys())}')
    return res