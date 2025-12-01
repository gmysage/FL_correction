"""
This module is functions specific to correct the self-absorption problem in 3D fluorescent tomography
"""
import os.path

import numpy as np
import h5py
import itertools
import xraylib
import time
from scipy import ndimage
from numba import jit, njit, prange, cuda
from .numba_util import *
from .image_util import *
from .util import *
from skimage import io
from tqdm import tqdm, trange
from multiprocessing import Pool, cpu_count
from functools import partial
from math import sin, cos, ceil, floor, pi
from skimage.transform import rescale
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    torch_available = True
except:
    torch_available = False

global mask3D

def load_global_mask(Mask3D):
    global mask3D
    mask3D = Mask3D

def load_global_atten_slices(atten_slices):
    global Atten_slices
    Atten_slices = atten_slices

def load_global_projection_slices(i_slices):
    global I_slices
    I_slices = i_slices    


def maximum_likelihood(img2D, p, y, n_iter=10):

    """
    Using maxium-likelihood methode to solve equation p*C = y in tomography reconstruction.

    Parameters:
    -----------
    img2D: 2D array
        initial guess of a 2D image
    A: 2D array
        general Radon transform matrix for the 2D image
    y: 1D array
        pixel value in the projection line.
    num: int
       number of iteration

    Returns:
    --------
    2D array:
        reconstructed 2D image with same size as img2D
    """

    img2D = np.array(img2D)
    img2D[np.isnan(img2D)] = 0

    A_old = img2D.flatten()    # convert 2D array in 1d array
    A_new = A_old
    for n in range(n_iter):
        print(f'iteration: {n}')
        Pf = p @ A_old
        Pf[Pf < 1e-6] = 1
        for j in range(np.size(A_old)):
            t1 = p[:,j]
            t2 = np.squeeze(y) / Pf
            a_sum = t1 @ t2
            b_sum = np.sum(p[:,j])
            if b_sum == 0:
                #b_sum = 1
                continue
            A_new[j] = A_old[j] * a_sum / b_sum
        A_old = A_new
        A_old[np.isnan(A_old)] = 0
    img_cor = np.reshape(A_old, img2D.shape)
    return img_cor


def mlem_matrix(img2D, p, y, n_iter=10):
    '''
    CPU version
    '''

    A_new = img2D.flatten().astype(np.float32)    # convert 2D array in 1d array
    A_new = A_new.reshape((len(A_new),1))
    b_sum = row_sum(p)     #b_sum = np.sum(p, axis=0)
    y[y<0] = 0
    if len(y.shape) == 1:
        y = y.reshape((len(y), 1))
    for _ in trange(n_iter):
        Pf = p @ A_new
        Pf[Pf < 1e-6] = 1
        t1 = p
        t2 = y / Pf
        a_sum = t2.T @ t1    
        A_new = A_new * (a_sum / b_sum).T
        A_new[np.isnan(A_new)] = 0
    img_cor = np.reshape(A_new, img2D.shape)
    return img_cor

def torch_mlem(img2D, p, y, n_iter, print_flag=True):
    if not torch_available:
        print('torch is not found')
        return 0
    #img2D = torch.tensor(img2D, dtype=torch.float)

    if print_flag:
        range_func = trange
    else:
        range_func = range
    A_new = img2D.flatten()
    A_new = A_new.reshape((len(A_new), 1))
    b_sum = torch.sum(p, axis=0, keepdim=True)

    if len(y.shape) == 1:
        y = y.reshape((len(y), 1))
    for _ in range_func(n_iter):
        Pf = torch.matmul(p, A_new)
        Pf = Pf.reshape((len(Pf), 1))
        Pf[Pf < 1e-6] = 1
        t1 = p
        t2 = y / Pf
        a_sum = t2.T @ t1
        A_new = A_new * (a_sum / b_sum).T
        A_new[torch.isnan(A_new)] = 0
    img_cor = A_new.reshape(img2D.shape)
    return img_cor

def torch_mlem_recon(sino_prj, angle_list, n_mlem_iter=200,
                     em_cs_ele=None, atten=None, img3D=None, pix=1, rho=1,
                     print_flag=True, device='cuda'):
    """
    sino_prj:  (n_angle, n_sli, 80)
    em_cs_ele: emission coefficient
    atten: attenuation coefficent (n_angle, 80, 80)
    img3D: initial guess (n_sli, 80, 80)

    NOTE:
        for regular tomography without attenuation, set atten=None
        if atten is given, it only works for reconstruction on single slice,
        as attenuaion will vary from slice to slice, and it will not be handled here
    """
    n_angle = len(angle_list) # 60

    s_prj = sino_prj.shape
    if len(s_prj) == 2:
        sino_prj = sino_prj[:, np.newaxis]
    s_prj = sino_prj.shape # e.g., (n_angle, 80, 80)

    s2 = (s_prj[-1], s_prj[-1])  # (80, 80)
    n_sli = s_prj[1]

    if img3D is None:
        img3D = np.ones((n_sli, *s2))
    s3 = img3D.shape
    if len(s3) == 2:
        img3D = img3D[np.newaxis]
    s3 = img3D.shape # (n_sli, 80, 80)

    if atten is None:
        atten = np.ones((n_angle, *s2))
    if em_cs_ele is None:
        em_cs_ele = np.ones(n_angle)
    theta_list = angle_list / 180. * np.pi
    H_tot = np.zeros((n_angle * s2[-1], s2[-1] * s2[-1]))
    H_zero = np.zeros((s2[-1], s2[-1] * s2[-1]))

    H = generate_H_jit(s3, theta_list, atten, H_tot, H_zero) # use update attenuation from ML recon
    H_comb = scale_expand_H_with_emission_coef(H, em_cs_ele)
    H_comb = H_comb * rho * pix
    rec = np.zeros(s3)
    for i in range(n_sli):
        sino2D = sino_prj[:, i]
        if np.max(sino2D) < 1e-8: # assume sinogram is 0
            continue
        I = sino2D.squeeze().flatten()
        I = I[:, np.newaxis]
        y = torch.tensor(I, dtype=torch.float).to(device)
        p = torch.tensor(H_comb, dtype=torch.float).to(device)
        img2D_torch = torch.tensor(img3D[i], dtype=torch.float32).to(device)
        img_cor = torch_mlem(img2D_torch, p, y, n_mlem_iter, print_flag).cpu().numpy()
        img_cor[img_cor<0] = 0
        rec[i] = img_cor
    rec = rec.squeeze()
    return rec


def torch_wpb_recon(img2D_torch, H, I, n_iter):
    s = img2D_torch.shape  # (80, 80)
    s_H = H.shape  # (4800, 6400)
    if torch.max(img2D_torch) == 0:
        C = torch.ones((s[0] * s[1], 1))
    else:
        C = img2D_torch.flatten()
        C = C.reshape((len(C), 1))  # (6400, 1)
    I = I.reshape((len(I), 1))  # e.g. (4800, 1)
    W = H.clone()
    C_new = C.clone()

    for i in trange(n_iter):
        COR = 0
        HC = H @ C_new  # (4800, 1)
        I_err = I - HC
        W = W * C_new.T
        '''    
        ### this is equivalent to W = W * C.T
        for r in range(s_H[0]):
            W[r] = W[r] * C[:, 0]
        '''
        ratio = I_err / torch.sum(W, axis=1, keepdim=True)
        ratio[torch.isnan(ratio)] = 0
        for r in range(s_H[0]):
            # cor = W[r] * I_err[r] / torch.sum(W[r])
            cor = W[r] * ratio[r]
            # cor[torch.isnan(cor)] = 0
            cor = cor.reshape((len(cor), 1))
            COR += cor
        C_new = COR + C_new

        C_new[C_new < 0] = 0
        # C[torch.isnan(C)] = 0
    img_new = C_new.reshape(s).cpu().numpy()
    plt.figure();
    plt.imshow(img_new);
    plt.title(f'n_iter={n_iter}')
    return img_new

def row_sum(p):
    s = p.shape
    m = np.ones((1, s[0]))
    p_sum = m @ p
    return p_sum

def prep_detector_mask3D(alfa=15, beta=60, length_maximum=200, fn_save='mask3D.h5'):
    print('Generating detector 3D mask ...')
    mask = {}
    for i in trange(length_maximum, 6, -1):
        mask[f'{i}'] = generate_detector_mask(alfa, beta, i)
    print(f'Saving {fn_save} ...')
    with h5py.File(fn_save, 'w') as hf:
        for i in range(7, length_maximum+1):
            k = f'{i}'
            hf.create_dataset(k, data=mask[k])
    return mask
    
    
def load_mask3D(fn='mask3D.h5'):
    f = h5py.File(fn, 'r')
    keys = f.keys()
    mask3D = {}
    for k in keys:
        mask3D[k] = np.array(f[k])
    return mask3D

def extract_single_value_param(param, energy_angle_id, dict_use_ref={}):
    elem_type = param['elem_type']
    elem_comp = param['elem_compound']
    param_angle_energy = param.copy()

    cs = param['cs']
    cs_angle_energy = deepcopy(cs)
    cs_angle_energy_type = type(cs_angle_energy[f'{elem_comp[0]}-x'])
    if cs_angle_energy_type is np.ndarray or list:
        for ele_comp in elem_comp:
            if len(cs[f'{ele_comp}-x']) == 1:
                cs_angle_energy[f'{ele_comp}-x'] = cs[f'{ele_comp}-x']
            else:
                cs_angle_energy[f'{ele_comp}-x'] = cs[f'{ele_comp}-x'][energy_angle_id]

    em_cs = param['em_cs']
    em_cs_angle_energy = deepcopy(em_cs)
    em_cs_angle_energy_type = type(em_cs_angle_energy[f'{elem_type[0]}'])

    if em_cs_angle_energy_type is np.ndarray or list:
        for ele in elem_type:
            if len(em_cs[ele]) == 1:
                em_cs_angle_energy[ele] = em_cs[ele]
            else:
                em_cs_angle_energy[ele] = em_cs[ele][energy_angle_id]

    """
    if reference spectrum provided, will compose the absorption cs with reference and chemical 
    concentration, e.g., dict_use_ref['ref_3D'] = (Ni2, Ni3), 
    Here, we only pick up the cs of reference at individual energy. 
    Note that id_angle == id_energy

    It pass the 'dict_use_ref' to function 'cal_atten_3D' to do real calculation 
    """
    if len(dict_use_ref) > 0:
        if dict_use_ref['use_ref']:  # True of False
            # update absorption cross-section
            for k in cs_angle_energy.keys():
                if 'x_ref' in k:
                    cs_angle_energy[k] = cs[k][energy_angle_id]
            # update emission cross-section
            for k in em_cs_angle_energy.keys():
                if ('x_ref' in k) and (not 'n_ref' in k):
                    em_cs_angle_energy[k] = em_cs[k][energy_angle_id]

    param_angle_energy['cs'] = cs_angle_energy
    param_angle_energy['em_cs'] = em_cs_angle_energy
    return param_angle_energy


def re_projection(img3D, angle_list, ax=1, order=3):
    """
    Project the 3D image along axis = ax after rotating the angle in the angle_list

    Parameters:
    -----------

    img3D:      3D array
    angle_list: 1D array
    ax:         rotation axis
                ax = 1 --> sum up the row (default)--> x-ray path is paralle to image's vertical direction
                ax = 2 --> sum up the colum

    Returns:
    --------

    3D array,
        e.g., if returns img_prj with shape of [10,200,200]:
              10 equals the number of angles in angle_list
              200 x 200 is the projection image size
    """

    img3D = np.array(img3D)
    img3D = rm_nan(img3D)
    im_size = img3D.shape
    ang_size = angle_list.shape

    if ax == 1:  # sum up the row --> projection from "top" to "bottom"
        img_prj = np.zeros([ang_size[0], im_size[0], im_size[2]])
        for i in range(ang_size[0]):
            print(f'current angle: {angle_list[i]}')
            img_prj[i] = np.sum(rot3D(img3D, angle_list[i], order), ax)

    elif ax == 2:  # sum up the colume --> projection from "left" to "right"
        img_prj = np.zeros([ang_size[0], im_size[0], im_size[1]])
        for i in range(ang_size[0]):
            print(f'current angle: {angle_list[j]}')
            img_prj[i] = np.sum(rot3D(img3D, angle_list[j], order), ax)
    else:
        print('check "ax". Nothing done with the image!')
        return img3D
    return img_prj

def re_projection_with_emission_and_attenuation(img3D, angle_list, param=None, rho_compound=1,
                                                atten_coef=None, elem='Ni', use_ref=False):
    '''
    calculate the xrf emission of the 2D projection from a 3D volume at all angles
    param contains the mass density (rho), pixel size (pix) and emission cross-section (em_cs)

    xrf(pixel) = concentration(pixel) * cs * rho * pixel_size

    if use_ref = True, it will read emission cross-section value from:
        param['em_cs']['Ni_ref1']
        param['em_cs']['Ni_ref2']

        img3D = (Ni_ref1_3D, Ni_ref2_3D)
    else:
        img3D = Ni_3D
    '''
    if param is None:
        prj = re_projection(img3D, angle_list, ax=1)
    else:
        pix = param['pix']
        em_cs = param['em_cs']
        n = len(angle_list)
        if atten_coef is None:
            atten_coef = np.ones(n)
        if use_ref==False:# regular
            s = img3D.shape
            prj = np.zeros((n, s[0], s[2]))
            for i_angle in trange(n):
                img3D_r = rot3D(img3D * rho_compound, angle_list[i_angle])
                coef = em_cs[elem][i_angle] * pix
                img_xrf = img3D_r * atten_coef[i_angle] * coef
                prj[i_angle] = np.sum(img_xrf, axis=1)
        else:
            n_ref = em_cs['n_ref'] # number of ref: e.g in em_cs: 'Ni_ref1', 'Ni_ref2'
            s = img3D[0].shape # e.g., 100 x 100 x 100
            prj = np.zeros((n, s[0], s[2]))
            for i_angle in trange(n):
                img_xrf = 0
                for i_ref in range(n_ref):
                    em_ref = em_cs[f'{elem}_ref{i_ref+1}']
                    img_xrf += em_ref[i_angle] * rho_compound * pix * img3D[i_ref]
                xrf_r = rot3D(img_xrf, angle_list[i_angle])
                xrf_r = xrf_r * atten_coef[i_angle]
                prj[i_angle] = np.sum(xrf_r, axis=1)
    return prj

def generate_detector_mask(alfa0, theta0, leng0):

    """
    Generate a pyramid shape mask inside a rectangular box matrix.

    Simulating the light transmission from a point source and then collected by rectangular detector

    Parameters:
    -----------

    alfa0:  int
            horizontal dispersion angle, in unit of degree

    theta0: int
            vertial dispersion angle, in unit of degree

    leng0:  int
            radial length of light transmission, in unit of pixels

    Returns:
    --------

    3D array:
        mask profile; matrix elements are zero outside the detection region.
    """

    alfa = np.float32(alfa0) / 2 / 180 * pi
    theta = np.float32(theta0) / 2 / 180 * pi

    N1_0 = np.int16(np.ceil(leng0 * np.tan(alfa))) # for original matrix
    N2_0 = np.int16(np.ceil(leng0 * np.tan(theta)))

    leng = leng0 + 30
    N1 = np.int16(np.ceil(leng * np.tan(alfa)))
    N2 = np.int16(np.ceil(leng * np.tan(theta)))

    Mask = np.zeros([2*N1-1, 2*N2-1, leng])

    s = Mask.shape
    s0= (2*N1_0-1, 2*N2_0-1, leng0) # size of "original" matrix

    M1 = np.zeros((s[0], s[2]))
    #M11 = M1.copy()
    M2 = np.zeros((s[1], s[2]))
    #M22 = M2.copy()

    M1 = g_mask((s[0], s[2]), alfa, N1)
    M2 = g_mask((s[1], s[2]), theta, N2)
    M1[N1-1,:] = 1
    M1[N1-1,0] = 0
    M2[N2-1,:] = 1
    M2[N2-1,0] = 0
    
    Mask1 = Mask.copy()
    Mask2 = Mask.copy()

    for i in range(s[1]):
        Mask1[:,i,:] = M1
    for i in range(s[0]):
        Mask2[i,:,:] = M2

    Mask = Mask1 * Mask2 # element by element multiply
    M_normal = g_radial_mask_approximate(Mask)

    '''
    # a more accurate calculation of "M_normal"

    shape_mask = np.int16(Mask > 0)
    a,b,c = np.mgrid[1:s[0]+1, 1:s[1]+1, 1:s[2]+1]
    dis = np.sqrt((a-N1)**2 + (b-N2)**2 + (c-1)**2)
    dis[N1-1,N2-1,0]=1
    dis = dis * shape_mask * 1.0
    M_normal = g_radial_mask(Mask.shape, dis, shape_mask) # generate mask with radial distance
    '''
    Mask3D = M_normal * Mask

    cent = np.array([np.floor(s[0]/2), np.floor(s[1]/2)])
    delt = np.array([np.floor(s0[0]/2), np.floor(s0[1]/2)])

    xs = np.int16(cent[0]-delt[0])
    xe = np.int16(cent[0]+delt[0]+1)
    ys = np.int16(cent[1]-delt[1])
    ye = np.int16(cent[1]+delt[1]+1)
    zs = 0
    ze = leng0

    Mask3D_cut = Mask3D[xs:xe,ys:ye,zs:ze] # col, slice, row
    Mask3D_cut = np.transpose(Mask3D_cut, [1,2,0]) # slice, row, col
    Mask3D_cut[np.isnan(Mask3D_cut)] = 0

    return Mask3D_cut

@njit
def g_mask(M_shape, alfa, N):
    tan_alfa = np.tan(alfa)
    cos_alfa = np.cos(alfa)

    M1 = np.zeros((M_shape[0], M_shape[1]))
    M11 = M1.copy()
    for I in range(M_shape[0]):
        for J in range(M_shape[1]):
            i = I+1
            j = J+1
            if (np.abs(N-i) >= j*tan_alfa):
                M1[I,J] = 0
                M11[I,J] = M1[I,J]

            elif (np.abs(N-i) < (j-1)*tan_alfa
                            and (np.abs(N-i)+1) > j*tan_alfa):

                desi_1 = (j-1)*tan_alfa - np.floor((j-1)*tan_alfa)
                desi_2 = j*tan_alfa - np.floor(j*tan_alfa)
                M11[I,J] = 0.5 * (desi_1 + desi_2)
                M1[I,J] = M11[I,J] / cos_alfa

            elif (np.abs(N-i) < j*tan_alfa
                            and (np.abs(N-i)+1) > j*tan_alfa
                            and np.abs(N-i) > (j-1)*tan_alfa):

                desi_1 = j*tan_alfa - np.floor(j*tan_alfa)
                M11[I,J] = 0.5 * desi_1 * (desi_1/tan_alfa)
                M1[I,J] = M11[I,J] / cos_alfa

            elif((np.abs(N-i)+1) < j*tan_alfa
                        and np.abs(N-i) < (j-1)*tan_alfa
                        and (np.abs(N-i)+1) > (j-1)*tan_alfa):
                desi_1 = np.ceil((j-1)*tan_alfa) - (j-1)*tan_alfa
                M11[I,J] = 1 - 0.5 * desi_1 * (desi_1/tan_alfa)
                M1[I,J] = M11[I,J] / cos_alfa

            else:
                tmp = np.arctan(1.0*(N-i)/j)
                M11[I,J] = 1
                M1[I,J] = M11[I,J] / np.cos(tmp)
    return M1  

@njit
def g_radial_mask(M_shape, radial_dis, shape_mask):

    M_normal = np.zeros(M_shape)
    dis = np.floor(radial_dis)
    a = np.floor(radial_dis)
    b = radial_dis - a
    for i in prange(1, np.max(dis)+1):
        #flag_mask = np.zeros(M_shape) # mark the position with radial distance == i
        temp = np.zeros(M_shape)

        s = dis.shape
        for p in prange(s[0]):
            for q in prange(s[1]):
                for r in prange(s[2]):
                    if dis[p, q, r] == i:
                        temp[p, q, r] = temp[p, q, r] + shape_mask[p, q, r]*(1-b[p, q, r])
                        #flag_mask[p, q, r]=1

                    if dis[p, q, r] == i-1:
                        temp[p, q, r] = temp[p, q, r] + shape_mask[p, q, r]*b[p, q, r]
                        #flag_mask[p, q, r]=1
        temp = 1.0 * temp / np.sum(temp)
        M_normal = M_normal + temp
    return M_normal


def g_radial_mask_approximate(Mask):
    s = Mask.shape
    M_normal = np.zeros(s)
    dis_sum = np.sum(np.sum(Mask, axis=0), axis=0)
    for k in range(s[-1]):
        M_normal[:, :, k] = Mask[:, :, k] / dis_sum[k]
    M_normal = rm_nan(M_normal)
    return M_normal

def get_mask_area_data(mask, data, row, col, sli):

    """
    Retrieve a block of 3D data from "data" defined by a mask profile "mask"
    Parameter of (rol, col, sli) control the starting point of the data to retrieve
    Be careful that the block should not exceed the shape boundary of data


    Parameters:
    -----------
    mask: 3D array
        mainly use the shape of "mask":
        [s0,s1,s2]=mask.shape

    data: 3D array

    row: int
        retrieve the data from row:row+s1
    col: int
        retrieve the data from (col-s2/2):(col+s2/2)
    sli: int
        retrieve the data from (sli-s0/2):(sli+s0/2)

    Returns:
    --------
    3D array
        retrieved data defined by mask, shape of data_maks equals shape of mask
    """

    s = mask.shape
    sd = data.shape

    xs = row
    xe = row + s[1]
    if xe > sd[1]: xe = sd[1]+1

    ys = col - int(np.floor(s[2] / 2))
    ye = ys + s[2]
    if ys < 0: ys = 0
    if ye > sd[2]: ye = sd[2]+1

    zs = sli - int(np.floor(s[0] / 2))
    ze = zs + s[0]
    if zs < 0: zs = 0
    if ze > sd[0]: ze = sd[0]+1

    data_mask = data[ze:ze, xs:xe, ys:ye]

    return data_mask


def edge_det_mask(img):

    """
    Generate a mask outline the edge of image using sobel kernal, and fill holes inside

    Parameters:
    -----------
    img: 2D array

    Returns:
    --------
    2D array: binary image
    """

    im = ndimage.filters.median_filter(img,7)
    im[im < 0.1*np.max(im)] = 0
    dx = ndimage.sobel(im, 1)
    dy = ndimage.sobel(im, 0)

    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)

    mag[mag < mag.mean()/2] = 0
    mag[mag > 0] = 1
    bw = mag

    str1 = ndimage.generate_binary_structure(2,2)
    bw_dl = ndimage.morphology.binary_dilation(bw, structure=str1, iterations=2).astype(bw.dtype)
    bw_fill = ndimage.morphology.binary_fill_holes(bw_dl).astype(bw_dl.dtype)
    bw_erode = ndimage.morphology.binary_erosion(bw_fill, structure=str1, iterations=3).astype(bw_fill.dtype)

    return bw_erode


def fit_surf(img, order=2):

    """
    Emperically fit a 2D data and then normalize it by the fitting surface(curve)
    e.g. img_ref, img_fit = fit_surf(img)

    Parameters:
    -----------

    img: 2D array
    order: int
        orders of polynomial equation

    Returns:
    --------
    z_fit: 2D array
        referece image from fitting
    im_fit: 2D array
        fitted image = raw_image / z_fit

    2D array

    """

    edge_mask = edge_det_mask(img)
    im = edge_mask * img

    na, nb = im.shape
    a = np.linspace(0, 1, na)
    b = np.linspace(0, 1, nb)
    av, bv = np.meshgrid(b, a)

    x = av.flatten()
    y = bv.flatten()
    z = im.flatten()

    mask = np.ones(z.size, dtype=bool)
    mask[z == 0] = False
    x = x[mask]
    y = y[mask]
    z = z[mask]

    coef = polyfit2d(x, y, z, order=order)
    z_fit = polyval2d(av, bv, coef)
    im_fit = img / z_fit


    im_fit = im_fit / np.max(im_fit) * np.max(im)

    im_fit = rm_nan(im_fit)
    return np.array(z_fit), np.array(im_fit)

def polyfit2d(x, y, z, order=2):

    """
    Fit z(x,y) using polynomial with order of 2(default).
    For example, a = polyfit2d(x, y, z, order=2) menas:
    z = a0 + a1*y + a2*y**2 + a3*x + a4*x*y + a5*x*y**2 + a6*x**2 + a7*x**2*y + a8*x**2*y**2

    Parameters:
    -----------

    x, y, z: 1d array or 2d array

    order: int

    Returns:
    --------

    1d or 2d array:
        coefficient of fitting, can be used for "polyval2d"
        E.g., z = polyval2d(x, y, coeff)

    """
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    ncols = (order + 1)**2
    A = np.zeros([x.size, ncols])
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        A[:, k] = x**i * y**j
    coef,_,_,_ = np.linalg.lstsq(A, z)
    # equavalen to (const, y, y^2, x, xy, xy^2, x^2, x^2y, x^2y^2)
    return coef


def polyval2d(x, y, coef):

    """
    Evaluate the value of z(x,y) with coefficient got from polyfit2d
    E.g., coeff = poly2dfit(x, y, z, order=3)
              c = polyval2d(a, b, coeff)

    Parameters:
    -----------

    x, y: 1d array or 2d array generated from numpy.meshgrid.

    coef: 1d or 2d array
        generated from polyfit2d

    Returns:
    --------

    1d or 2d array:
        fitted curve/surface, data has same shape as x (or y)

    """
    order = int(np.sqrt(len(coef))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(coef, ij):
        z += a * x**i * y**j
    return z



def write_attenuation(elem, data, angle_id, file_path='./Angle_prj'):

    """
    write attenuation coefficient into file in the directory of './Angle_prj' as:

    'atten_gd_prj_-45.h5'

    where:  elem = 'gd'
            current_angle = -45

    """

    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root  = file_path
    else:
        fn_root  = file_path + '/'
    fname = fn_root + 'atten_' + elem + '_prj_' + f'{angle_id:04d}' + '.tiff'
    io.imsave(fname, data.astype(np.float32))

def write_attenuation_fl(elem, data, angle_id, file_path='./Angle_prj'):

    """
    write fluorescence attenuation coefficient for element:

    'atten_gd_prj_-45.h5'

    where:  elem = 'gd'
            current_angle = -45

    """

    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root  = file_path
    else:
        fn_root  = file_path + '/'
    fname = fn_root + 'atten_fl_' + elem + '_prj_' + f'{angle_id:04d}' + '.tiff'
    io.imsave(fname, data.astype(np.float32))

def write_attenuation_xray(elem, data, angle_id, file_path='./Angle_prj'):

    """
    write incident x-ray attenuation coefficient for element:

    'atten_gd_prj_0000.h5'

    where:  elem = 'gd'
            angle_id = 0

    """

    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root  = file_path
    else:
        fn_root  = file_path + '/'
    fname = fn_root + 'atten_incident_xray_' + elem + '_prj_' + f'{angle_id:04d}' + '.tiff'
    io.imsave(fname, data.astype(np.float32))


def write_projection(mode, elem, data, current_angle, angle_id, file_path='./Angle_prj'):

    """
    write aligned projection image into file with filename as:

    'gd_ref_prj_-45.h5'

    where:  elem = 'gd'
            current_angle = -45

    Parematers:
    -----------

    mode: chars, choosing from:
        "single_file" or "s" --> save as single projection file, then 'current_angle' is a anlge_list
        "multi_file" or "m"  --> save as multi (seperated) projection file, then 'current_angle' is simply a number

    data: 2D or 3D, depending on write into seperated projection file(2D), or a single combined file(3D)
    """

    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root = file_path 
    else:
        fn_root = file_path + '/'
    if mode == 'single_file' or mode == 's':
        fname = fn_root + elem + '_ref_prj_single_file.h5'
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('dataset_1', data=data)
            hf.create_dataset('angle_list', data=current_angle)
    elif mode == 'multi_file' or mode == 'm':
        fname = fn_root + elem + '_ref_prj_' + f'{angle_id:04d}' + '.tiff'
        io.imsave(fname, data.astype(np.float32))
    else: print ('unrecognized "mode"')


def pre_treat(*args):

    """
    Combine a list of 3D data(with same shape) to a 4D array.

    Parameters:
    -----------
    a list of 3D data, or one 4D array

    Returns:
    --------
    4D array, and a array consist of the shape of 3D data

    """

    n = len(args)
    if n == 1:                                    # one set of 4D or 3D data
        data = np.squeeze(np.array(args))
        if len(data.shape) == 4:
            data = rm_nan(data)
            return np.array(data)
        elif len(data.shape) == 3:
            data = np.zeros([1]+list(data.shape))
            data[0] = args[0]
            data = rm_nan(data)
            return np.array(data)
        else:
            print('input should be either 3D or 4D data')
            return 0

    if n > 1:                                     # a list of 3D or 2D data
        s = args[0].shape
        ss = [n] + [s[i] for i in range(len(s))]
        data = np.zeros(ss)
        for i in range(n):
            data[i] = args[i]
        data = rm_nan(data)
        return np.array(data)

def get_atten_coef(elem_type, elem_compound, XEng_list, em_E):
    '''
    elem_type = ['Zr', 'La', 'Hf']
    elem_compound = ['ZrO2', 'La2O3', 'HfO2']
    XEng = [12]
    em_E = [4.6, ]
    '''
    et = elem_type
    ec = elem_compound
    cs = {}
    n = len(elem_type)
    cs['elem_type'] = elem_type
    cs['elem_compound'] = elem_compound
    cs['x'] = XEng_list # incident x-ray energy
    for i in range(n):
        # atten at incident x-ray
        cs[f'{ec[i]}-x'] = np.zeros(len(XEng_list))
        for k in range(len(XEng_list)):
            XEng = XEng_list[k]
            # cs[f'{et[i]}-x'][k] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(et[i]), XEng)
            cs[f'{ec[i]}-x'][k]  = xraylib.CS_Total_CP(ec[i], XEng)
        for j in range(n):
            # atten at each emission line
            # cs[f'{et[i]}-{et[j]}'] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(et[i]), em_E[et[j]])
            cs[f'{ec[i]}-{et[j]}'] = xraylib.CS_Total_CP(ec[i], em_E[et[j]])
    return cs



def cal_atten_with_direction(img4D_r, param_angle_energy, position_det='r', enable_scale=False,
                             detector_offset_angle=0, dict_use_ref_angle={}, num_cpu=8):
    """
    Calculate the attenuation of incident x-ray, fluorescent with given
    experiment configuration. Assume x-ray is passing from front to the back
    of the 3D object

    Rotate the 3D volume to change configuration to:
    x-ray: change to pass from left->rigth of the 3D volume
    detector: sit in front of the 3D volume, which means it sit at bottom of each 2D slice of 3D 

    call sub-routine of "cal_atten_3D"

    Parameters:
    -----------

    img4D: dimension of (num_of_element, [shape of 3D data])

    param: paramters loaded from files

    cs: cross-section values derived from parameter

    position_dec: detector position
        'r': right side of 3D object
        'l': left side of 3D object
        default: 'r'

    return:
    -----------
    total Attenuation

    fluorescent Attenuation

    incident x-ray Attenuation
    """

    elem_type = param_angle_energy['elem_type']
    n_type = img4D_r.shape[0]
    
    # change detector position from "right (or left) to "front"    
    if position_det == 'r': # if detector locates on the right side:
        img4D_rr = fast_rot90_4D(img4D_r, ax=0, mode='clock')
    if position_det == 'l': # if detector locates on the left side:
        img4D_rr = fast_rot90_4D(img4D_r, ax=0, mode='c-clock')
        img4D_rr = img4D_rr[:, :, :, ::-1]

    # need to rotate dict_use_ref['ref_3D']
    dict_use_ref_rotate = deepcopy(dict_use_ref_angle)
    if len(dict_use_ref_rotate) > 0:
        if dict_use_ref_rotate['use_ref']:
            ref_3D = dict_use_ref_rotate['ref_3D']
            n_ref = len(ref_3D)
            if position_det == 'r': # if detector locates on the right side:
                for i in range(n_ref):
                    dict_use_ref_rotate['ref_3D'][i] = fast_rot90_3D(ref_3D[i], ax=0, mode='clock')

            if position_det == 'l': # if detector locates on the left side:
                for i in range(n_ref):
                    ref_3D[i] = fast_rot90_3D(ref_3D[i], ax=0, mode='c-clock')
                    dict_use_ref_rotate['ref_3D'][i] = ref_3D[i][:, :, ::-1]

    """
    Note, in modified "cal_atten_3D", 
    atten_xray is element specific: atten_xray['Ni', 'Co', 'Mn'] 
    the total x-ray attenuation will be: atten_xray['Ni'] * atten_xray['Mn'] * atten_xray['Co']    
    """
    atten, atten_fl, atten_xray = cal_atten_3D(img4D_rr, param_angle_energy, enable_scale=enable_scale,
                                               detector_offset_angle=detector_offset_angle,
                                               dict_use_ref_rotate=dict_use_ref_rotate,
                                               num_cpu=num_cpu)


    atten3D = {}
    atten3D_fl = {}
    atten3D_xray = {}
    # reverse the rotation 
    for i in range(n_type):
        ele = elem_type[i]
        if position_det == 'r': 
            tmp = fast_rot90_3D(atten[ele], ax=0, mode='c-clock')
            atten3D[ele] = tmp

            tmp1 = fast_rot90_3D(atten_fl[ele], ax=0, mode='c-clock')
            atten3D_fl[ele] = tmp1

            #tmp2 = fast_rot90_3D(atten_xray, ax=0, mode='c-clock')
            tmp2 = fast_rot90_3D(atten_xray[ele], ax=0, mode='c-clock')
            atten3D_xray[ele] = tmp2
        if position_det == 'l': 
            tmp = atten[ele][:, :, ::-1]
            tmp = fast_rot90_3D(tmp, ax=0, mode='clock')
            atten3D[ele] = tmp

            tmp1 = atten_fl[ele][:, :, ::-1]
            tmp1 = fast_rot90_3D(tmp1, ax=0, mode='clock')
            atten3D_fl[ele] = tmp1

            #tmp2 = atten_xray[:, :, ::-1]
            tmp2 = atten_xray[ele][:, :, ::-1]
            tmp2 = fast_rot90_3D(tmp2, ax=0, mode='clock')
            atten3D_xray[ele] = tmp2
    return atten3D, atten3D_fl, atten3D_xray



def generate_H(elem, sli, angle_list, bad_angle_index=[], fpath_atten='./Angle_prj', flag=1, shape_2D=[]):
    """
    Generate matriz H and I for solving eqution: H*C=I
    In folder of file_path:
            Needs 3D attenuation matrix at each rotation angle
    e.g. H = generate_H('Gd', Gd_tomo, 30, angle_list, bad_angle_index, file_path='./Angle_prj', flag=1)

    Parameters:
    -----------

    elem_type: chars
        e.g. elem_type='Gd'
    ref3D_tomo: 3d array
        a referenced 3D tomography data with same shape of attentuation matrix
    sli: int
        index of slice ID in 3D tomo
    angle_list: 1d array
        rotation angles in unit of degree
    bad_angle_index: 1d array
        angle_index that angle will not be used,
        e.g. bad_angle_index=[0,10,36] --> angle_list[0] will not be used
    file_path: folder path. Under the path, it includes:
        attenuation matrix with name of:  e.g. atten_Gd_prj_50.0.h5
        these files can be generated through function of: 'write_attenuation'
    flag: int
        flag = 1: use attenuation matrix read from file
        flag = 0: generate non-attenuated matrix (Radon matrix)

    Returns:
    --------
    2D array
    """
    n = len(angle_list)
    if flag: # read from file
        atten_3D = []
        for i in range(n):
            att = read_attenuation(i, fpath_atten, elem)[sli]
            atten_3D.append(att)
        atten_3D = np.array(atten_3D)
    else:
        s = shape_2D
        atten_3D = np.ones((n, *s))
    theta_list = np.array(angle_list / 180 * np.pi)
    atten_3D = exclude_idx(atten_3D, bad_angle_index)
    theta_list = exclude_idx(theta_list, bad_angle_index)
    H_tot = generate_H_from_atten_coef(atten_3D, theta_list)
    return H_tot

def generate_H_from_atten_coef(atten_list, theta_list):
    '''
    atten_list = (60, 100, 100) --> (n_angles, n_pix, n_pix)
    theta_list (60, ): in unit if radians, if not, will convert it automatically
    '''
    if np.max(np.abs(theta_list)) > 2 * np.pi:
        theta_list = theta_list / 180. * np.pi
    num = len(theta_list)
    s = atten_list.shape # (60, 80, 80)

    H_tot = np.zeros([s[2] * num, s[2] * s[2]])

    cx = (s[2]-1) / 2.0       # center of col
    cy = (s[1]-1) / 2.0

    for i in trange(len(theta_list)):
        att = atten_list[i]
        theta1 = theta_list[i]
        T = np.array([[np.cos(-theta1), -np.sin(-theta1)],[np.sin(-theta1), np.cos(-theta1)]])
        H = np.zeros([s[2], s[1] * s[2]])
        for col in range(s[2]):
            for row in range(s[1]):
                p = row
                q = col
                cord = np.dot(T,[[p-cx],[q-cy]]) + [[cx],[cy]]
                if ((cord[0] > s[1]-1) or (cord[0] < 0) or (cord[1] > s[2]-1) or (cord[1] < 0)):
                    continue
                r_frac = cord[0] - np.floor(cord[0])
                c_frac = cord[1] - np.floor(cord[1])
                r_up = int(np.floor(cord[0]))
                r_down = int(np.ceil(cord[0]))
                c_left = int(np.floor(cord[1]))
                c_right = int(np.ceil(cord[1]))

                ul = r_up * s[2] + c_left
                ur = r_up * s[2] + c_right
                dl = r_down * s[2] + c_left
                dr = r_down * s[2] + c_right

                if (r_up >= 0 and c_left >=0):
                    H[q, ul] = H[q, ul] + att[p, q] * (1-r_frac) * (1-c_frac)
                if (c_left >=0):
                    H[q, dl] = H[q, dl] + att[p, q] * r_frac * (1-c_frac)
                if (r_up >= 0):
                    H[q, ur] = H[q, ur] + att[p,q] * (1-r_frac) * c_frac
                H[q, dr] =  H[q, dr] + att[p, q] * r_frac * c_frac
        H_tot[i * s[2]: (i + 1) * s[2], :] = H
    return H_tot

def generate_I(elem, ref3D_tomo, sli, angle_list, bad_angle_index=[], file_path='./Angle_prj'):

    """
    Generate matriz I for solving eqution: H*C=I
    In folder of file_path:
            Needs aligned 2D projection at each rotation angle
    e.g. I = generate_I(Gd, 30, angle_list, bad_angle_index, file_path='./Angle_prj')

    Parameters:
    -----------

    elem: chars
        e.g. elem='Gd'
    ref3D_tomo: 3d array
        a referenced 3D tomography data with same shape of attentuation matrix
    sli: int
        index of slice ID in 3D tomo
    angle_list: 1d array
        rotation angles in unit of degree
    bad_angle_index: 1d array
        angle_index that angle will not be used,
        e.g. bad_angle_index=[0,10,36] --> angle_list[0] will not be used
    file_path: folder path. Under the path, it includes:
         aligned projection image:         e.g. Gd_ref_prj_50.0.h5
         these files can be generated through function of: 'write_projection'

    Returns:
    --------
    1D array

    """

    #print('Reding aligned projection and generate matrix "I"...')

    # fpre_prj = file_path + '/' + elem_type.lower() + '_ref_prj_'
    fpre_prj = file_path + '/' + elem + '_ref_prj_'
    theta = np.array(angle_list / 180 * np.pi)
    num = len(theta) - len(bad_angle_index)
    s = ref3D_tomo.shape
    n = s[2]*num
    I_tot = np.zeros(s[2]*num)
    k = -1
    for i in range(len(theta)):
        if i in bad_angle_index:
            continue
        k = k + 1
        f_ref = fpre_prj + f'{i:04d}.tiff'
        prj = io.imread(f_ref)[sli]
        I_tot[k*s[2] : (k+1)*s[2]] = prj
    return I_tot


def generate_I_slices(elem, img3D_shape, angle_list, bad_angle_index=[], file_path='./Angle_prj'):

    #print('Reding aligned projection and generate matrix "I"...')

    # fpre_prj = file_path + '/' + elem_type.lower() + '_ref_prj_'
    fpre_prj = file_path + '/' + elem + '_ref_prj_'
    theta = np.array(angle_list / 180 * np.pi)
    n_ang = len(theta) - len(bad_angle_index)
    s = img3D_shape
    n_sli = s[0]
    n = s[2]*n_ang
    I_slices = np.zeros((n_sli, n))
    k = -1
    for i in trange(len(theta)):
        if i in bad_angle_index:
            continue
        k = k + 1
        f_ref = fpre_prj + f'{i:04d}.tiff'
        prj = io.imread(f_ref)
        for sli in range(s[0]): # slices
            I_slices[sli, k*s[2] : (k+1)*s[2]] = prj[sli]
    return I_slices


def scale_expand_H_with_emission_coef(H_tot, em_cs_ref):
    s = em_cs_ref.shape # e.g., (60, 2) or (60,)
    if len(s) == 1:
        em_cs_ref = em_cs_ref.reshape((s[0], 1))
    s = em_cs_ref.shape
    n_angle = s[0]
    n_ref = s[1]
    if n_angle==1 and n_ref==1: # a single value of em_cs_ref:
        H_comb = H_tot * em_cs_ref
    else:
        s0 = H_tot.shape # e.g., (60x80, 80x80) --> image size = (80,80), n_angle=60
        n_pix = int(np.sqrt(s0[1])) # 80
        H_comb = np.zeros((s0[0], s0[1]*n_ref))
        for i in range(n_angle):
            rs = i * n_pix
            re = (i + 1) * n_pix
            for j in range(n_ref):
                cs = j * s0[1]
                ce = (j + 1) * s0[1]
                H_comb[rs:re, cs:ce] = H_tot[rs:re] * em_cs_ref[i, j]
    return H_comb



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


def atten_2D_slice(sli, mu_ele):
    global mask3D
    s_mu = mu_ele.shape
    mu_max = np.max(mu_ele)
    atten_ele = np.ones([s_mu[1], s_mu[2]])
    #sli = int(s_mu[0]/2)
    for i in range(s_mu[1]): # row
        length = max(s_mu[1] - i, 7)
        mask = np.asarray(mask3D[f'{length}'], dtype='f4')
        s_mask = mask.shape
        for j in np.arange(0, s_mu[2]): # column
            cord = retrieve_data_mask_cord(s_mu, s_mask, sli, int(i), int(j))
            zs, ze, xs, xe, ys, ye, m_zs, m_ze, m_xs, m_xe, m_ys, m_ye = cord
            sub_volume_mu = mu_ele[zs:ze, xs:xe, ys:ye]
            sub_volume_mask = mask[m_zs:m_ze, m_xs:m_xe, m_ys:m_ye]
            atten_ele[i, j] = np.sum(sub_volume_mu * sub_volume_mask)
            #atten_ele[i, j] = retrieve_data_mask(mu_ele, int(i), int(j), sli, mask)
    return atten_ele


def atten_2D_slice_all_elem(sli):
    '''
    mu_all_elem: for all elements, shape = (5, 75, 100, 100)
    '''
    global mu_all_elem
    s_mu = mu_all_elem.shape  # (5, 75, 100, 100)
    n_elem, n_sli, n_row, n_col = s_mu
    atten_all = np.ones((n_elem, n_row, n_col)) # (5, 100, 100)
    for i in range(n_row): # row
        length = max(n_row-i, 7)
        mask = np.asarray(mask3D[f'{length}'], dtype='f4')
        s_mask = mask.shape
        for j in np.arange(0, n_col): # column
            cord = retrieve_data_mask_cord(s_mu[1:], s_mask, sli, int(i), int(j))
            zs, ze, xs, xe, ys, ye, m_zs, m_ze, m_xs, m_xe, m_ys, m_ye = cord  
            for k in range(n_elem):
                sub_volume_mu = mu_all_elem[k, zs:ze, xs:xe, ys:ye]
                sub_volume_mask = mask[m_zs:m_ze, m_xs:m_xe, m_ys:m_ye]
                mu_masked = sub_volume_mu * sub_volume_mask
                atten_all[k, i, j] = np.sum(mu_masked)   
            
    return atten_all


def retrieve_data_mask_cord(data_shape, mask_shape, sli, row, col):
    s0 = data_shape
    s = mask_shape
    xs = int(row)
    xe = int(row + s[1])
    ys = int(col - floor(s[2]/2))
    ye = int(col + floor(s[2]/2)+1)
    zs = int(sli - floor(s[0]/2))
    ze = int(sli + floor(s[0]/2)+1)
    ms = mask_shape
    m_xs = 0   
    m_xe = ms[1]

    m_ys = 0
    m_ye = ms[2]

    m_zs = 0
    m_ze = ms[0]
    if xs < 0:
        m_xs = -xs
        xs = 0
    if xe > s0[1]:
        m_xe = s0[1] - xe + ms[1]
        xe = s0[1]
    if ys < 0:
        m_ys = -ys
        ys = 0
    if ye > s0[2]:
        m_ye = s0[2] - ye + ms[2]
        ye = s0[2]
    if zs < 0:
        m_zs = -zs
        zs = 0
    if ze > s0[0]:
        m_ze = s0[0] - ze + ms[0]
        ze = s0[0]
    cord = [zs, ze, xs, xe, ys, ye, m_zs, m_ze, m_xs, m_xe, m_ys, m_ye]
    return cord

def atten_2D_slice_mpi(sli, mu_ele, num_cpu=4):
    max_num_cpu = round(cpu_count() * 0.8)
    if num_cpu == 0 or num_cpu > max_num_cpu:
        num_cpu = max_num_cpu
    print(f'assembling {len(sli)} slices using {num_cpu:2d} CPUs')
    partial_func = partial(atten_2D_slice, mu_ele=mu_ele)
    pool = Pool(num_cpu)
    #res = pool.map(partial_func, sli)
    res = []
    for result in tqdm(pool.imap(func=partial_func, iterable=sli), total=len(sli)):
        res.append(result)
    pool.close()
    pool.join()
    res = np.array(res)
    #res = process_map(atten_2D_slice, sli, mu_ele, False, max_worder=num_cpu)
    return res


def atten_voxel(idx, row, mu_ele):
    #global mask3D
    s = mu_ele.shape
    length = max(s[1] - row, 7)
    mask = np.asarray(mask3D[f'{length}'], dtype='f4')

    n_sli = s[0]
    s_col = s[2]
    sli = idx // n_sli
    col = idx - sli * n_sli
    atten = retrieve_data_mask(mu_ele, row, col, sli, mask)
    return atten


def atten_2D_row_mpi(row, mu_ele, display_flag=True, num_cpu=4):

    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    #from tqdm.contrib.concurrent import process_map
    from functools import partial
    max_num_cpu = round(cpu_count() * 0.8)
    if num_cpu == 0 or num_cpu > max_num_cpu:
        num_cpu = max_num_cpu
    #print(f'assembling row using {num_cpu:2d} CPUs')
    s = mu_ele.shape
    n_sli = s[0]
    n_col = s[2]
    idx = np.arange(n_sli*n_col)
    partial_func = partial(atten_voxel, row=row, mu_ele=mu_ele)
    pool = Pool(num_cpu)
    #res = pool.map(partial_func, sli)
    res = []
    for result in tqdm(pool.imap(func=partial_func, iterable=idx), total=len(idx)):
        res.append(result)
    pool.close()
    pool.join()
    res = np.array(res).reshape((n_sli, n_col))
    return res


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

"""
def cal_frac_simple(*args):
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
    img_sum = np.sum(img, axis=0)

    pix_max = np.max(img_sum)
    img_mask = adaptive_threshold(img_sum, fill_hole=True, dilation=1)
    img_sum = img_sum * img_mask

    sum_sort = np.sort(img_sum[img_sum > (pix_max) * 0.1])
    pix_max_update = sum_sort[int(0.95 * len(sum_sort))]
    pix_median = np.median(sum_sort)

    for i in range(n):
        frac[i]  = rm_abnormal(img[i] / img_sum)
    res = {}
    res['img_sum'] = img_sum
    res['frac'] = frac
    res['pix_median'] = pix_median
    res['pix_max'] = pix_max_update
    res['img_mask'] = img_mask
    return res
"""

@jit
def retrieve_data_mask(data3d, row, col, sli, mask):
    """
    Retrieve data defined by mask, orignated at position(sli/2, row, col/2),
    and then multiply it by mask, and then take the sum

    Parameters:
    -----------
    data: 3d array

    row, col, sli: int
          (sli/2, row, col/2) is the original of the position to retrieve data

    mask: 3d array
          shape of mask should be smaller than shape of data

    Returns:
    --------
    3D array:  data defined by mask-shape multiplied by mask

    """
    s0 = data3d.shape
    s = mask.shape
    xs = int(row)
    xe = int(row + s[1])
    ys = int(col - floor(s[2]/2))
    ye = int(col + floor(s[2]/2)+1)
    zs = int(sli - floor(s[0]/2))
    ze = int(sli + floor(s[0]/2)+1)
    ms = mask.shape
    m_xs = 0   
    m_xe = ms[1]

    m_ys = 0
    m_ye = ms[2]

    m_zs = 0
    m_ze = ms[0]
    if xs < 0:
        m_xs = -xs
        xs = 0
    if xe > s0[1]:
        m_xe = s0[1] - xe + ms[1]
        xe = s0[1]
    if ys < 0:
        m_ys = -ys
        ys = 0
    if ye > s0[2]:
        m_ye = s0[2] - ye + ms[2]
        ye = s0[2]
    if zs < 0:
        m_zs = -zs
        zs = 0
    if ze > s0[0]:
        m_ze = s0[0] - ze + ms[0]
        ze = s0[0]
    data = np.sum(data3d[zs:ze, xs:xe, ys:ye] * mask[m_zs:m_ze, m_xs:m_xe, m_ys:m_ye])
    return data


def cal_atten_3D(img4D_rr, param_angle_energy, enable_scale=False,
                 detector_offset_angle=0, dict_use_ref_rotate={}, num_cpu=8):
    """
    Geometry:
        XRF Detector is in front of 3D volume (h, r, c) --> it is at the end of row (r)
        incident X-ray is on the left of 3D volume (h, r, c) --> it is at the beginning of column (c)

    special note:
    detector_offset_angle: angle of detector away from 90 degrees to incident xray
    e.g. 0: detector is at 90 deg to incident x-ray
        15: detortor is at 75 deg to incident x-ray, in this case, need to c-clock rotate data to calculate xrf-atten
    """

    elem_type = param_angle_energy['elem_type']
    #res = cal_frac(img4D_r, scale_range=[0.1, 0.9])
    res = cal_frac(img4D_rr, scale_range=[0.02, 0.98], enable_scale=enable_scale, scale_limit=0.95)
    frac = res['frac']
    atten_fl = {} # single-pix xrf attenuation coef of each element
    atten = {}

    # xrf attenuation
    atten_xrf = cal_xrf_atten(frac, param_angle_energy, detector_offset_angle, num_cpu)
    for i, elem in enumerate(elem_type):
        atten_fl[elem] = atten_xrf[i]
    """
    # incident x-ray attenuation
    #x_ray_atten = cal_incident_x_ray_atten(frac, param_angle_energy, dict_use_ref_rotate)
    
    atten_overall = atten_xrf * x_ray_atten  # (xrf + incident xray) attenuation

    # group into elements
    for i, elem in enumerate(elem_type):
        atten_fl[elem] = atten_xrf[i]
        atten[elem] = atten_overall[i]
    """
    # version v2 for incident x-ray attenuation (grouped in to elements)
    # e.g., x_ray_atten_elem = {'Ni', 'Co', 'Mn'}
    x_ray_atten_elem = cal_incident_x_ray_atten_v2(frac, param_angle_energy, dict_use_ref_rotate)
    x_ray_atten = 1
    for elem in elem_type:
        x_ray_atten = x_ray_atten * x_ray_atten_elem[elem]
    for i, elem in enumerate(elem_type):
        atten[elem] = x_ray_atten * atten_fl[elem]

    # return atten, atten_fl, x_ray_atten
    return atten, atten_fl, x_ray_atten_elem


"""
# Note: 
function: "xrf_atten" has been replaced with "cal_xrf_atten"
def xrf_atten(dict_mu, elem_type, num_cpu=1):
    '''
    Geometry:
        Detector is in front of 3D volume (h, r, c), it is at the end of row (r)

    mu_all_elem, 4D array, shape = (5, 75, 100, 100)
    num_cpu:
        if 1: run in sequence
        if >1: run using mpi
    '''
    global mu_all_elem
    mu_all_elem = pre_treat(list(dict_mu[k] for k in elem_type))
    max_num_cpu = round(cpu_count() * 0.8)
    if num_cpu == 0 or num_cpu > max_num_cpu:
        num_cpu = max_num_cpu

    s = mu_all_elem.shape # (5, 75, 100, 100)
    n_sli = s[1]
    slices = np.arange(n_sli)

    print(f'assembling {len(slices)} slices using {num_cpu:2d} CPUs')
    ts = time.time()
    #partial_func = partial(atten_2D_slice_all_elem, mu_all_elem=mu_all_elem)
    partial_func = atten_2D_slice_all_elem
    pool = Pool(num_cpu)
    res = []
    for result in tqdm(pool.imap(func=partial_func, iterable=slices), total=n_sli):
        res.append(result)
    pool.close()
    pool.join()
    res = np.array(res) # (75, 5, 100, 100)
    res = np.transpose(res, (1, 0, 2, 3)) # (5, 75, 100, 100)
    te = time.time()
    print(te-ts)
    return res
"""


def cal_incident_x_ray_atten(frac_4D, param_angle_energy, dict_use_ref_rotate={}):
    """
    img4D: [n_elem, 100, 100, 100]

    Note: incident x-ray is from the "left" side of the 3D volume

    dict_use_ref: parameter determine if use reference spectrum to interpolate absorption cross-section
        dict_use_ref['use_ref'] = True
        dict_use_ref['ref_3D'] = (Ni2, Ni3)
        dict_use_ref['ref_comp'] = 'LiNiO2'
    """
    if len(dict_use_ref_rotate) > 0:
        use_ref = dict_use_ref_rotate['use_ref']
        ref_3D = dict_use_ref_rotate['ref_3D']
        ref_comp = dict_use_ref_rotate['ref_comp']
    else:
        use_ref = False
    cs_ = deepcopy(param_angle_energy['cs'])
    elem_compound = cs_['elem_compound']
    rho = param_angle_energy['rho']
    pix = param_angle_energy['pix']
    frac = frac_4D
    mu = {'x':0}
    if use_ref:
        assert ref_comp is not None, "need to provide 'ref_comp', e.g, 'LiNiO2'"
        cs_ref = compose_cs_incident_xray_with_reference(ref_3D, cs_, ref_comp)
        cs_[f'{ref_comp}-x'] = cs_ref # this is a 3D volume of cross-section

    for j, elem_comp in enumerate(elem_compound):
        # e.g
        # mu['x'] = mu[8.4 keV] = f[LiNiO2] * cs[LiNiO2-8.4keV] * rho[LiNiO2] +
        #                       + f[LiCoO2] * cs[LiCoO2-8.4keV] * rho[LiCoO2] +
        #                       + f[LiMnO2] * cs[LiMnO2-8.4keV] * rho[LiMnO2] +
        # note if use ref:
        # cs[LiNiO2-8.4keV] is a 3D volume
        # cs[LiCoO2-8.4keV] and is cs[LiMnO2-8.4keV] is single value
        mu['x'] += frac[j] * cs_[f'{elem_comp}-x'] * rho[elem_comp]

    s = frac_4D[0].shape # (100, 100, 100)
    x_ray_atten = np.ones(s)
    for j in range(1, s[2]):
        # x_ray_atten[:, :, j] = x_ray_atten[:, :, j-1] * np.exp(-mu['x'][:, :, j-1] * pix)
        # since mu['x'][:, :, j-1] * pix is very small, e.g. for 100nm (1e-5cm)pix, and mu=1000, the value is 0.01
        # exp(-mu['x'][:, :, j-1] * pix) ~ 1-mu['x'][:, :, j-1] * pix

        t = mu['x'][:, :, j - 1] * pix
        #t = np.exp(-mu['x'][:, :, j-1] * pix)
        x_ray_atten[:, :, j] = x_ray_atten[:, :, j - 1] * (1-t)
    return x_ray_atten


def cal_incident_x_ray_atten_v2(frac_4D, param_angle_energy, dict_use_ref_rotate={}):
    """
    img4D: [n_elem, 100, 100, 100]

    Note: incident x-ray is from the "left" side of the 3D volume

    dict_use_ref: parameter determine if use reference spectrum to interpolate absorption cross-section
        dict_use_ref['use_ref'] = True
        dict_use_ref['ref_3D'] = (Ni2, Ni3)
        dict_use_ref['ref_comp'] = 'LiNiO2'
    """
    if len(dict_use_ref_rotate) > 0:
        use_ref = dict_use_ref_rotate['use_ref']
        ref_3D = dict_use_ref_rotate['ref_3D']
        ref_comp = dict_use_ref_rotate['ref_comp']
    else:
        use_ref = False
    cs_ = deepcopy(param_angle_energy['cs'])
    elem_type = cs_['elem_type']
    elem_compound = cs_['elem_compound']
    rho = param_angle_energy['rho']
    pix = param_angle_energy['pix']
    frac = frac_4D
    mu = {}
    if use_ref:
        assert ref_comp is not None, "need to provide 'ref_comp', e.g, 'LiNiO2'"
        cs_ref = compose_cs_incident_xray_with_reference(ref_3D, cs_, ref_comp)
        cs_[f'{ref_comp}-x'] = cs_ref # this is a 3D volume of cross-section

    for j, elem_comp in enumerate(elem_compound):
        # e.g
        # mu['x'] = mu[8.4 keV] = f[LiNiO2] * cs[LiNiO2-8.4keV] * rho[LiNiO2] +
        #                       + f[LiCoO2] * cs[LiCoO2-8.4keV] * rho[LiCoO2] +
        #                       + f[LiMnO2] * cs[LiMnO2-8.4keV] * rho[LiMnO2] +
        # note if use ref:
        # cs[LiNiO2-8.4keV] is a 3D volume
        # cs[LiCoO2-8.4keV] and is cs[LiMnO2-8.4keV] is single value
        ele = elem_type[j]
        mu[ele] = frac[j] * cs_[f'{elem_comp}-x'] * rho[elem_comp]
        #mu['x'] += frac[j] * cs_[f'{elem_comp}-x'] * rho[elem_comp]

    s = frac_4D[0].shape # (100, 100, 100)
    x_ray_atten = {}
    for ele in elem_type:
        x_ray_atten[ele] = np.ones(s)
        for j in range(1, s[2]):
            # x_ray_atten[:, :, j] = x_ray_atten[:, :, j-1] * np.exp(-mu['x'][:, :, j-1] * pix)
            # since mu['x'][:, :, j-1] * pix is very small, e.g. for 100nm (1e-5cm)pix, and mu=1000, the value is 0.01
            # exp(-mu['x'][:, :, j-1] * pix) ~ 1-mu['x'][:, :, j-1] * pix

            t = mu[ele][:, :, j - 1] * pix
            #t = np.exp(-mu['x'][:, :, j-1] * pix)
            x_ray_atten[ele][:, :, j] = x_ray_atten[ele][:, :, j - 1] * (1-t)
    return x_ray_atten


def cal_xrf_atten(frac_4D, param_angle_energy, detector_offset_angle=0, num_cpu=1):
    '''
        Geometry:
            Detector is in front of 3D volume (h, r, c), it is at the end of row (r)

        mu_all_elem, 4D array, shape = (5, 75, 100, 100)
        num_cpu:
            if 1: run in sequence
            if >1: run using mpi
        '''
    global mu_all_elem

    cs_ = deepcopy(param_angle_energy['cs'])
    elem_type = cs_['elem_type'] # 'Ni', 'Co', 'Mn'
    elem_compound = cs_['elem_compound'] # 'LiNiO2', 'LiCoO2', 'LiMnO2'
    rho = param_angle_energy['rho']
    pix = param_angle_energy['pix']
    frac = frac_4D
    dict_mu = {}
    for i, xrf_eng in enumerate(elem_type): # ['Ni', 'Co', 'Mn']
        dict_mu[xrf_eng] = 0
        for j, elem_comp in enumerate(elem_compound):
            # e.g
            # mu['Ni'] = mu[7.4keV] = f[LiNiO2] * cs[LiNiO2-7.4keV] * rho[LiNiO2] +
            #                       + f[LiCoO2] * cs[LiCoO2-7.4keV] * rho[LiCoO2] +
            #                       + f[LiMnO2] * cs[LiMnO2-7.4keV] * rho[LiMnO2] +
            dict_mu[xrf_eng] += frac[j] * cs_[f'{elem_comp}-{xrf_eng}'] * rho[elem_comp]

    mu_xrf = rot3D_dict_img(dict_mu, detector_offset_angle)
    mu_all_elem = pre_treat(list(mu_xrf[k] for k in elem_type))
    max_num_cpu = round(cpu_count() * 0.8)
    if num_cpu == 0 or num_cpu > max_num_cpu:
        num_cpu = max_num_cpu
    s = mu_all_elem.shape # (5, 75, 100, 100)
    n_sli = s[1]
    slices = np.arange(n_sli)
    print(f'assembling {len(slices)} slices using {num_cpu:2d} CPUs')
    ts = time.time()
    #partial_func = partial(atten_2D_slice_all_elem, mu_all_elem=mu_all_elem)
    partial_func = atten_2D_slice_all_elem
    pool = Pool(num_cpu)
    res = []
    for result in tqdm(pool.imap(func=partial_func, iterable=slices), total=n_sli):
        res.append(result)
    pool.close()
    pool.join()
    res = np.array(res) # (75, 5, 100, 100)
    res = np.transpose(res, (1, 0, 2, 3)) # (5, 75, 100, 100)
    res = rot3D(res, -detector_offset_angle)
    atten_xrf = np.exp(-res * pix)
    te = time.time()
    print(te-ts)
    return atten_xrf


def compose_cs_incident_xray_with_reference(ref_3D, ref_cs, ref_comp):
    '''
    ref_3D = (Ni2, Ni3), e.g., Ni2(3).shape=(100, 100, 100)
    ref_cs.keys = ['LiNiO2-x_ref1', 'LiNiO2-x_ref2', ...]
    ref_comp = 'LiNiO2'

    ref_cs: e.g., ['LiNiO2-x_ref1', 'LiNiO2-x_ref2']
        ref_cs['LiNiO2-x_ref1']: a single value, corresponding to the reference cross-section at single x-ray energy
        Note: 'ref_cs' is adapted from "cs_angle_energy" from function "cal_and_save_atten_prj"

    return:
    cs_4D: 3D cross-section at all energies, shape = (n_eng, 100, 100, 100)
    '''
    n_ref = ref_cs['n_ref']
    assert len(ref_3D) == n_ref, "number of references and number of 3D stack in 'ref_3D' does not match"
    ref_spec = np.zeros(n_ref)
    for i_ref in range(n_ref):
        ref_spec[i_ref] = ref_cs[f'{ref_comp}-x_ref{i_ref + 1}']
    ref_frac = cal_frac(ref_3D, scale_range=[], enable_scale=False)['frac']
    cs_3D = 0
    for i_ref in range(n_ref):
        cs_3D += ref_spec[i_ref] * ref_frac[i_ref]
    return cs_3D



@njit(parallel=True, fastmath=True)
def generate_H_jit(tomo3D_shape, theta, atten3D, H_tot, H_zero):

    """
    Generate matriz H and I for solving eqution: H*C=I
    In folder of file_path:
            Needs 3D attenuation matrix at each rotation angle
    e.g. H = generate_H('Gd', Gd_tomo, 30, angle_list, bad_angle_index, file_path='./Angle_prj', flag=1)

    Parameters:
    -----------

    elem_type: chars
        e.g. elem_type='Gd'
    ref3D_tomo: 3d array
        a referenced 3D tomography data with same shape of attentuation matrix
    theta: 1d array
        rotation angles in unit of radius

    Returns:
    --------
    2D array
    """

    # s = tomo3D_shape e.g., (200, 200, 200)
    # theta = np.array(angle_list / 180 * np.pi)
    # num = len(theta)
    # H_tot.shape = (n * s[2], s[1] * s[2])
    # H_zero.shape = (s[2], s[1] * s[2])
    num = len(theta)
    s = tomo3D_shape
    cx = (s[2]-1) / 2.0       # center of col
    cy = (s[1]-1) / 2.0       # center of row
    #H_tot = np.zeros([s[2]*num, s[2]*s[2]])
    H = H_zero * 0
    H_tot = H_tot * 0
    for i in prange(num):
        att = atten3D[i]
        #T = np.array([[np.cos(-theta[i]), -np.sin(-theta[i])],[np.sin(-theta[i]), np.cos(-theta[i])]])
        T00 = np.cos(-theta[i])
        T01 = -np.sin(-theta[i])
        T10 = np.sin(-theta[i])
        T11 = np.cos(-theta[i])
        H = H * 0
        for col in range(s[2]):
            for row in range(s[1]):
                p = row
                q = col
                #cord = np.dot(T,[[p-cx],[q-cy]]) + [[cx],[cy]]
                t0 = T00 * (p-cx) + T01 * (q-cy) + cx
                t1 = T10 * (p-cx) + T11 * (q-cy) + cy
                cord = [t0, t1]
                if ((cord[0] > s[1]-1) or (cord[0] < 0) or (cord[1] > s[2]-1) or (cord[1] < 0)):
                    continue
                
                r_frac = cord[0] - np.floor(cord[0])
                c_frac = cord[1] - np.floor(cord[1])
                r_up = int(np.floor(cord[0]))
                r_down = int(np.ceil(cord[0]))
                c_left = int(np.floor(cord[1]))
                c_right = int(np.ceil(cord[1]))

                ul = r_up * s[2] + c_left
                ur = r_up * s[2] + c_right
                dl = r_down * s[2] + c_left
                dr = r_down * s[2] + c_right

                if (r_up >= 0 and c_left >=0):
                    H[q, ul] = H[q, ul] + att[p, q] * (1-r_frac) * (1-c_frac)
                if (c_left >=0):
                    H[q, dl] = H[q, dl] + att[p, q] * r_frac * (1-c_frac)
                if (r_up >= 0):
                    H[q, ur] = H[q, ur] + att[p,q] * (1-r_frac) * c_frac
                H[q, dr] =  H[q, dr] + att[p, q] * r_frac * c_frac
        H_tot[i*s[2] : (i+1)*s[2], :] = H

    return H_tot



def smooth_filter(img, filter_size=3):
    s = img.shape
    img_s = np.zeros(s)
    n = s[0]
    for i in range(n):
        #img_smooth[i] = ndimage.gaussian_filter(img[i], 1.1)
        img_s[i] = ndimage.median_filter(img[i], 3)
    return img_s


def cal_and_save_atten_prj(param, recon4D, angle_list, ref_prj, fsave='./Angle_prj',
                           align_flag=0, enable_scale=True, detector_offset_angle=0,
                           dict_use_ref={}, num_cpu=8):
    '''
    Function used to calcuate the attenuation coefficent of each element and all 3D voxels
    results will be saved into the "fsave" folder

    If align_flag = 1:
        it will do image alignment using the forwared projection image as reference, and align
    the raw projection image ("ref_prj": the projection image collected from experiment)

    If align_flag = 0:
        it will simply copy the raw projection image into the folder, which assuming the raw projection images
        is well alinged with rotation center locted at image center.

    dict_use_ref = {}
        dict_use_ref['use_ref'] = True
        dict_use_ref['ref_comp'] = 'LiNiO2'
        dict_use_ref['ref_3D'] = (Ni2, Ni3)
        If use_ref = True:
            need to provide ref_3D:
                e.g., ref_3D = pre_treat(Ni2, Ni3), Ni2(3): concentration of Ni2+ and Ni3+, shape = (100, 100, 100)
            need to provide ref_comp:
                e.g., 'LiNiO2', should be one of the compound listed in param['elem_compound']
            assume cs has keys of ['LiNiO2-x_ref1', 'LiNiO2-x_ref2', ...'

            it will re-calculate the absorption cross-section (cs) according to ['LiNiO2-x_ref1', 'LiNiO2-x_ref2']
            it generate a 4D cross-section e.g., (n_energy, 100, 100, 100)
    '''
    if ref_prj is None:
        write_prj = False
    else:
        write_prj = True
    s = recon4D.shape
    elem_type = param['elem_type']
    Nelem = len(elem_type)
    n_angle = len(angle_list)
    if write_prj:
        prj = np.zeros([Nelem, n_angle, s[1], s[3]])
        ref_prj_sum = np.sum(ref_prj, axis=0)
    for ang_id in range(n_angle):
        print(f'\ncalculate and save attenuation and projection at angle {angle_list[ang_id]}: {ang_id+1}/{n_angle}')

        param_angle_energy = extract_single_value_param(param, ang_id, dict_use_ref)

        res = cal_atten_prj_at_angle(angle_list[ang_id], recon4D, param_angle_energy, position_det='r',
                                     enable_scale=enable_scale, detector_offset_angle=detector_offset_angle,
                                     dict_use_ref=dict_use_ref, num_cpu=num_cpu)

        if align_flag:
            _, r, c = align_img(res['prj_sum'], ref_prj_sum[ang_id])     
            print(f'shift projection image at angle={angle_list[ang_id]}: {r}, {c}')   
        for i in range(Nelem):
            elem = elem_type[i]
            if write_prj:
                if align_flag:
                    prj[i, ang_id] = ndimage.shift(ref_prj[i, ang_id], [r, c])
                else:
                    prj[i, ang_id] = ref_prj[i, ang_id]
                write_projection('m', elem, prj[i, ang_id], angle_list[ang_id], ang_id, fsave)
            write_attenuation(elem, res['atten'][elem], ang_id, fsave)
            # write fluorecent attenuantion
            write_attenuation_fl(elem, res['atten_fl'][elem], ang_id, fsave)
            # write incident xray attenuantion
            write_attenuation_xray(elem, res['atten_xray'][elem], ang_id, fsave)


def cal_atten_prj_at_angle(current_angle, img4D, param_angle_energy, position_det='r', enable_scale=False,
                           detector_offset_angle=0, dict_use_ref={}, num_cpu=8):
    '''
    calculate the attenuation and projection at single angle
    '''

    elem_type = param_angle_energy['elem_type']
    n_type = len(elem_type)
    img4D_r = rot3D(img4D, current_angle)
    prj = {}
    prj_sum = 0

    dict_use_ref_angle = deepcopy(dict_use_ref)
    # need to rotate dict_use_ref['ref_3D']
    if len(dict_use_ref_angle) > 0:
        if dict_use_ref_angle['use_ref']:
            ref_3D = dict_use_ref_angle['ref_3D']
            n_ref = len(ref_3D)
            for i in range(n_ref):
                dict_use_ref_angle['ref_3D'][i] = rot3D(ref_3D[i], current_angle)


    atten3D, atten3D_fl, atten3D_xray = cal_atten_with_direction(img4D_r, param_angle_energy, position_det=position_det,
                                                           enable_scale=enable_scale,
                                                           detector_offset_angle=detector_offset_angle,
                                                           dict_use_ref_angle=dict_use_ref_angle,
                                                           num_cpu=num_cpu)
    for j in range(n_type):
        ele = elem_type[j]
        prj[ele] = np.sum(img4D_r[j]*atten3D[ele], axis=1)
        prj_sum = prj_sum + prj[ele]
    res = {}
    res['atten'] = atten3D
    res['prj'] = prj
    res['prj_sum'] = prj_sum
    res['atten_fl'] = atten3D_fl
    res['atten_xray'] = atten3D_xray
    return res



def absorption_correction_mpi(elem, ref_tomo, angle_list, fpath_atten, n_iter, 
                                num_cpu=4, save_tiff=True, fpath_save='./recon'):

    global Atten_slices
    atten4D = read_attenuation_at_all_angle(angle_list, fpath_atten, elem)
    Atten_slices = np.transpose(atten4D, [1, 0, 2, 3]) # (300, 53, 400, 400)
    n_sli = Atten_slices.shape[0]
    slices = np.arange(n_sli)

    partial_func = partial(absorption_correction, elem=elem,
                        ref_tomo=ref_tomo, angle_list=angle_list,
                        fpath_atten=fpath_atten, n_iter=n_iter,
                        save_tiff=save_tiff, fpath_save=fpath_save)
    pool = Pool(num_cpu)
    res = []
    ts = time.time()
    
    for result in tqdm(pool.imap(func=partial_func, iterable=slices), total=len(slices)):
        res.append(result)
    '''
    for result in tqdm(pool.imap(func=partial_func, iterable=zip(sli, atten_4d)), total=len(sli)):
        res.append(result)
    '''
    pool.close()
    pool.join()
    te = time.time()
    print(f'take {te-ts:2.1f} sec')
    res = np.array(res)
    return res


def absorption_correction(sli_id, elem, ref_tomo, angle_list, fpath_atten, n_iter=10, 
                            save_tiff=True, fpath_save='./recon'):
    global Atten_slices
    # Atten_slices.shape = (300, 53, 400, 400)
    I_tot = generate_I(elem, ref_tomo, sli_id, angle_list, [], fpath_atten)    
    
    # angle_id = np.arange(len(angle_list))
    #coef_att4D = read_attenuation_at_all_angle(fpath_atten, elem, angle_list) # iter=-10 --> rescale attenuation coefficient to original size
    #atten3D = coef_att4D[:, sli_id]
    
    '''
    s = coef_att4D.shape # (53, 300, 400, 400)
    tomo_shape = s[1:]
    H_tot = np.zeros((s[0]*s[3], s[2]*s[3]))
    H_zero = np.zeros((s[3], s[2]*s[3]))
    '''
    atten3D = Atten_slices[sli_id]
    s = atten3D.shape # (53, 400, 400) # re-aranged attenuation coefficient for single slice at all angles
    tomo3D_shape = (1, *s[1:])
    H_tot = np.zeros((s[0]*s[2], s[1]*s[2])) # (53x400, 400*400)
    H_zero = np.zeros((s[2], s[1]*s[2])) # (400, 400*400)

    theta = angle_list / 180. * np.pi    
    H_jit = generate_H_jit(tomo3D_shape, theta, atten3D, H_tot, H_zero)
    img2D = ref_tomo[sli_id]
    img2D[img2D<0] = 0
    res = mlem_matrix(img2D, H_jit, I_tot, n_iter)
    if save_tiff:        
        if fpath_save[-1] == '/':
            fpath_save = fpath_save[:-1]
        fsave_root = fpath_save + f'/{elem}'
        mk_directory(fsave_root)
        fsave = fsave_root + f'/{elem}_slice_{sli_id:04d}.tiff'
        io.imsave(fsave, res.astype(np.float32))
    return res



def cuda_generate_H_single_slice(atten_sli, angle_list, H0=None, H_tot=None,
                                 blocks_2D=(16, 16), threads_2D=(16, 16)):
    '''
    using numba.cuda
    return H_tot as cuda_array
    '''
    #blocks_2D = (16, 16)
    #threads_2D = (16, 16)
    theta = angle_list / 180. * np.pi
    s = atten_sli.shape # e.g, (53, 400, 400) --> 53 angles
    tomo_shape = (1, *s[1:])
    if H0 is None or (not cuda.is_cuda_array(H0)):
        h0 = np.zeros((s[2], s[1]*s[2]), dtype=np.float32)
        H0 = cuda.to_device(h0)
    if H_tot is None or (not cuda.is_cuda_array(H_tot)):
        h_tot = np.zeros((s[0]*s[2], s[1]*s[2]), dtype=np.float32)
        H_tot = cuda.to_device(h_tot)
    for i in range(s[0]):
        theta_i = theta[i]
        att = atten_sli[i].astype(np.float32)
        att = cuda.to_device(att)
        kernel_generate_H_single_angle[blocks_2D, threads_2D](tomo_shape, theta_i, att, H0)
        cuda.synchronize()
        H_tot[i*s[2] : (i+1)*s[2], :] = H0
        kernel_zeros_2Darray[16, 16](H0)
        cuda.synchronize()
    return H_tot


def cuda_absorption_correction_single_slice(img_sli, I_sli, atten_sli, angle_list, 
                                            H0=None, H_tot=None, n_iter=10,
                                            save_tiff=False, fn_save='tmp.tiff'):
    '''
    recommend to have H0 and H_tot in cuda_array already
    img_sli: 
        2D numpy array, intital guess. E.g.,    shape = (400, 400)
    I_lis: 
        2D numpy array, flattened projection image at all angles. shape = (n, 1)
    atten_sli:
        3D (numpy / cuda) array, attenuation coefficient at all angles. shape = (n_angles, 400, 400)
    H0:
        2D (numpy / cuda) array
    H_tot:
        2D (numpy / cuda) array

    return: 2D numpy array 
    '''
    
    theta = angle_list / 180. * np.pi
    s = atten_sli.shape # e.g, (53, 400, 400) --> 53 angles
    tomo_shape = (1, *s[1:])
    if H0 is None or (not cuda.is_cuda_array(H0)):
        print('transfer H0 to cuda device')
        h0 = np.zeros((s[2], s[1]*s[2]), dtype=np.float32)
        H0 = cuda.to_device(h0)

    if H_tot is None or (not cuda.is_cuda_array(H_tot)):
        print('transfer H_tot to cuda device')
        h_tot = np.zeros((s[0]*s[2], s[1]*s[2]), dtype=np.float32)
        H_tot = cuda.to_device(h_tot)

    cuda_generate_H_single_slice(atten_sli, angle_list, H0, H_tot)
    
    if not cuda.is_cuda_array(I_sli):
        if len(I_sli.shape) == 1: # if it is single column vector, convert to (n, 1) array
            I_sli = I_sli.reshape((len(I_sli), 1))
        I_sli = cuda.to_device(I_sli.astype(np.float32))
    
    if not cuda.is_cuda_array(atten_sli):
        atten_sli = cuda.to_device(atten_sli.astype(np.float32))

    img_sli[img_sli<0] = 0
    x0 = img_sli.flatten().reshape((img_sli.size, 1))
    X = cuda.to_device(x0.astype(np.float32))

    mlem_cuda(X, H_tot, I_sli, n_iter)
    x = X.copy_to_host()
    x_new = x.reshape(img_sli.shape)
    if save_tiff:
        io.imsave(fn_save, x_new)
    return x_new


def cuda_absorption_correction_wrap(elem, ref_tomo, angle_list, fpath_atten, n_iter, save_tiff, fpath_save):
    img3D_shape = ref_tomo.shape
    Atten_slices, I_slices = prepare_atten_I(elem, img3D_shape, angle_list, fpath_atten)
    fpath_save_elem = fpath_save + f'/{elem}'
    mk_directory(fpath_save_elem)
    img_cor = cuda_absorption_correction_batch(ref_tomo, I_slices, Atten_slices, angle_list, n_iter, 
                                    save_tiff, fpath_save_elem)
    return img_cor                           

def cuda_absorption_correction_batch(img3D, I_slices, Atten_slices, angle_list, n_iter=10, 
                                    save_tiff=False, fpath_save_elem='.'):
    '''
    img3D: 
        3D numpy array, shape = (300, 400, 400)
    I_slices:
        2D numpy array, shape = (n_slices, len(I_sli))
    Atten_slices:
        4D numpy array, shape = (n_slices, n_angles, 400, 400), e.g, (300, 53, 400, 400)
    '''
    
    s_img = img3D.shape  # e.g., (300, 400, 400)
    n_sli = s_img[0]   # e.g., n_sli=300
    n_ang = len(angle_list) # e.g, n_ang=53

    s = (n_ang, *s_img[1:])
    h0 = np.zeros((s[2], s[1]*s[2]), dtype=np.float32)
    H0 = cuda.to_device(h0)
    h_tot = np.zeros((s[0]*s[2], s[1]*s[2]), dtype=np.float32)
    H_tot = cuda.to_device(h_tot)

    img_cor3D = np.zeros(s_img)
    for i in trange(n_sli):
        fn_save = fpath_save_elem + f'/img_{i:04d}.tiff' 
        img_sli = img3D[i]
        I_sli = I_slices[i]
        I_sli = I_sli.reshape((len(I_sli), 1))
        atten_sli = Atten_slices[i]
        img_cor3D[i] = cuda_absorption_correction_single_slice(img_sli, I_sli, atten_sli, angle_list, 
                                                                H0, H_tot, n_iter, save_tiff, fn_save)
        kernel_zeros_2Darray[(16, 16), (16, 16)](H0)
        cuda.synchronize()
        kernel_zeros_2Darray[(16, 16), (16, 16)](H_tot)
        cuda.synchronize()
    return img_cor3D


def prepare_atten_I(elem, img3D_shape, angle_list, fpath_atten):
    
    s_img = img3D_shape
    n_sli = s_img[0]
    n_ang = len(angle_list)

    atten4D = read_attenuation_at_all_angle(angle_list, fpath_atten, elem) # (n_ang, 300, 400,400)
    Atten_slices = np.transpose(atten4D, [1, 0, 2, 3]) # (300, 53, 400, 400)
    I_slices = generate_I_slices(elem, s_img, angle_list, bad_angle_index=[], file_path=fpath_atten)

    return Atten_slices, I_slices


def get_K_edge(elem='Ni'):
    return xraylib.EdgeEnergy(xraylib.SymbolToAtomicNumber(elem), 0)

def get_attenuation_length_K_edge(elem='Ni', compound='LiNiO2', rho=4.65):
    edge = get_K_edge(elem)
    e0 = edge - 0.1
    e1 = edge + 0.1
    eng_test = np.linspace(e0, e1, 1000)
    tmp_cs = np.zeros(1000)
    for i in range(1000):
        tmp_cs[i] = xraylib.CS_Total_CP(compound, eng_test[i])
    jump = np.max(tmp_cs) - np.min(tmp_cs)
    atten_length = 1.0 / jump / rho
    return atten_length

def read_and_rescale_atten_coef(proj_raw, fn_root, current_iter_id, elem_type, prj_prefix='Angle_prj', scale_factor=2, add_n_pix=[0,0,0]):
    fn_read_folder = fn_root + f'/{prj_prefix}_{current_iter_id:02d}'
    fn_save_folder = fn_root + f'/{prj_prefix}_-{current_iter_id:02d}'
    tmp = np.sort(glob.glob(fn_read_folder + f'/atten_fl_{elem_type[0]}_prj*'))
    n_angle = len(tmp)
    b = scale_factor
    for elem_id, elem in enumerate(elem_type):
        print(f'processing {elem} ... ')
        for i in trange(n_angle):
            coef_att = read_attenuation(i, fn_read_folder, elem)
            s = coef_att.shape
            s_new = s * b
            coef_scale = rescale(coef_att, b)
            coef_scale = pad_matrix_lines(coef_scale, add_n_pix)
            write_attenuation(elem, coef_scale, i, fn_save_folder)

            coef_fl = read_attenuation_fl(i, fn_read_folder, elem)
            coef_fl_scale = rescale(coef_fl, b)
            coef_fl_scale = pad_matrix_lines(coef_fl_scale, add_n_pix)
            write_attenuation_fl(elem, coef_fl_scale, i, fn_save_folder)

            coef_xray = read_attenuation_xray(i, fn_read_folder, elem)
            coef_xray_scale = rescale(coef_xray, b)
            coef_xray_scale = pad_matrix_lines(coef_xray_scale, add_n_pix)
            write_attenuation_xray(elem, coef_xray_scale, i, fn_save_folder)

            write_projection('m', elem, proj_raw[elem_id, i], i, i, fn_save_folder)

def pad_matrix_lines(data, n_pix=[0,0,0]):
    s = np.array(data.shape)
    n_pix = np.array(n_pix)
    if len(s) != len(n_pix):
        print('dimension disagree')
    else:
        s1 = s + n_pix
        d = np.zeros(s1)
        if len(s) == 2:
            d[:s[0], :s[1]] = data
            for r in range(s[0], s1[0]):
                d[r] = d[s[0]-1]
            for c in range(s[1], s1[1]):
                d[:, c] = d[:, s[1]-1]
        if len(s) == 3:
            d[:s[0], :s[1], :s[2]] = data
            for h in range(s[0], s1[0]):
                d[h] = d[s[0]-1]
            for r in range(s[1], s1[1]):
                d[:, r] = d[:, s[1]-1]
            for c in range(s[2], s1[2]):
                d[:, :, c] = d[:, :, s[2]-1]
        else:
            d = data
            print('Nothing has been done')
    return d

def prepare_ref_cs(param, ref_comp='LiNiO2', ref_elem='Ni'):
    n_eng = len(param['eng_list'])
    # absorption cross-section
    cs_ = param['cs']
    n_ref = cs_['n_ref']
    ref_cs = np.zeros((n_eng, n_ref))
    for i_ref in range(n_ref):
        ref_cs[:, i_ref] = cs_[f'{ref_comp}-x_ref{i_ref + 1}']
    # emission cross-section
    cs_emission = param['em_cs']
    ref_em_cs = np.zeros((n_eng, n_ref))
    for i_ref in range(n_ref):
        ref_em_cs[:, i_ref] = cs_emission[f'{ref_elem}_ref{i_ref + 1}']
    return ref_cs, ref_em_cs


def prepare_ref_cs_frac_obsolete(recon3D, param, ref_comp='LiNiO2', ref_elem='Ni'):
    '''
    recon3D: tuple, e.g., (Ni, Co, Mn), the first 1 conresponds to the element of interest, e.g. Ni

    '''
    n_elem = len(recon3D)
    s = recon3D[0].shape
    img3D = np.zeros((n_elem, *s))

    for i in range(n_elem):
        img3D[i] = recon3D[i]
    frac4D = cal_frac(img3D, scale_range=[])['frac']  # (3, 100, 100, 100)
    n_eng = len(param['eng_list'])

    # absorption cross-section
    cs_ = param['cs']
    n_ref = cs_['n_ref']
    ref_cs = np.zeros((n_eng, n_ref))
    for i_ref in range(n_ref):
        ref_cs[:, i_ref] = cs_[f'{ref_comp}-x_ref{i_ref + 1}']

    # emission cross-section
    cs_emission = param['em_cs']
    ref_em_cs = np.zeros((n_eng, n_ref))
    for i_ref in range(n_ref):
        ref_em_cs[:, i_ref] = cs_emission[f'{ref_elem}_ref{i_ref + 1}']

    return ref_cs, ref_em_cs, frac4D
##

