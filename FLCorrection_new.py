"""
This module is functions specific to correct the self-absorption problem in 3D fluorescent tomography
"""
import os
import numpy as np
import h5py
import copy
import itertools
import matplotlib.pyplot as pl
import xraylib
import scipy.ndimage
import time
from scipy import ndimage
from scipy.signal import medfilt
from numpy import pi, sin, cos, nan
from align_class import dftregistration
from numba import jit, njit, prange
from skimage import io
import warnings
warnings.filterwarnings('ignore')

global mask3D

def load_mask3D(fn='mask3D.h5'):
    f = h5py.File(fn, 'r')
    keys = f.keys()
    mask3D = {}
    for k in keys:
        mask3D[k] = np.array(f[k])
    return mask3D

def maximum_likelihood(img2D, p, y, iter_num=10):

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
    for n in range(iter_num):
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

def mlem_matrix(img2D, p, y, iter_num=10):
    img2D = np.array(img2D)
    img2D[np.isnan(img2D)] = 0
    img2D[img2D < 0] = 0

    A_new = img2D.flatten()    # convert 2D array in 1d array
    for n in range(iter_num):
        print(f'iteration: {n}')
        Pf = p @ A_new
        Pf[Pf < 1e-6] = 1
        t1 = p
        t2 = y.flatten() / Pf.flatten()
        t2 = np.reshape(t2, (len(t2), 1))
        a_sum = np.sum(t1*t2, axis=0)
        b_sum = np.sum(p, axis=0)
        a_sum[b_sum<=0] = 0
        b_sum[b_sum<=0] = 1
        A_new = A_new * a_sum / b_sum
        A_new[np.isnan(A_new)] = 0
    img_cor = np.reshape(A_new, img2D.shape)
    return img_cor

def pad4d(img_stack, thick=0, direction=0):
    """
    img_stack contians a set of 3d or 2d image
    direction: int
        0: padding in axes = 0 (2D or 3D image)
        1: padding in axes = 1 (2D or 3D image)
        2: padding in axes = 2 (3D image)
    """

    temp = pad(img_stack[0], thick=thick, direction=direction)
    data = np.zeros([img_stack.shape[0]] + list(temp.shape))
    data[0] = temp
    for i in range(img_stack.shape[0]-1):
        data[i+1] = pad(img_stack[i+1], thick=thick, direction=direction)

    return data



def pad(img, thick, direction):

    """
    symmetrically padding the image with "0"

    Parameters:
    -----------
    img: 2d or 3d array
        2D or 3D images
    thick: int
        padding thickness for all directions
        if thick == odd, automatically increase it to thick+1
    direction: int
        0: padding in axes = 0 (2D or 3D image)
        1: padding in axes = 1 (2D or 3D image)
        2: padding in axes = 2 (3D image)

    Return:
    -------
    2d or 3d array

    """

    thick = np.int32(thick)
    if thick%2 == 1:
        thick = thick + 1
        print(f'Increasing padding thickness to: {thick}')

    img = np.array(img)
    s = np.array(img.shape)
    if thick == 0 or direction > 3 or s.size > 3:
        return img

    hf = np.int32(np.ceil(abs(thick)+1) / 2)  # half size of padding thickness
    if thick > 0:
        if s.size < 3:  # 2D image
            if direction == 0: # padding row
                pad_image = np.zeros(s[0]+thick, s[1])
                pad_image[hf:(s[0]+hf), :] = img

            else:  # direction == 1, padding colume
                pad_image = np.zeros([s[0], s[1]+thick])
                pad_image[:, hf:(s[1]+hf)] = img

        else:  # s.size ==3, 3D image
            if direction == 0:  # padding slice
                pad_image = np.zeros([s[0]+thick, s[1], s[2]])
                pad_image[hf:(s[0]+hf), :, :] = img

            elif direction ==1:  # padding row
                pad_image = np.zeros([s[0], s[1]+thick, s[2]])
                pad_image[:, hf:(s[1]+hf), :] = img

            else:  # padding colume
                pad_image = np.zeros([s[0],s[1],s[2]+thick])
                pad_image[:, :, hf:(s[2]+hf)] = img

    else: # thick < 0: shrink the image
        if s.size < 3:  # 2D image
            if direction == 0:  # shrink row
                pad_image = img[hf:(s[0]-hf), :]

            else:  pad_image = img[:, hf:(s[1]-hf)]    # shrink colume

        else:  # s.size == 3, 3D image
            if direction == 0:  # shrink slice
                pad_image = img[hf:(s[0]-hf), :, :]

            elif direction == 1:  # shrink row
                pad_image = img[:, hf:(s[1]-hf),:]

            else:  # shrik colume
                pad_image = img[:, :, hf:(s[2]-hf)]
    return pad_image


def rot3D(img, rot_angle):

    """
    Rotate 2D or 3D or 4D(set of 3D) image with angle = rot_angle

    Parameters:
    -----------
    img:        2D or 3D array or 4D array

    rot_angle:  float
                rotation angles, in unit of degree

    Returns:
    --------
    2D or 3D or 4D array with same shape of input image
        all pixel value is large > 0

    """

    img = np.array(img)
    img = rm_nan(img)
    s = img.shape
    if len(s) == 2:    # 2D image
        img_rot = ndimage.rotate(img, rot_angle, reshape=False)
    elif len(s) == 3:  # 3D image, rotating along axes=0
        img_rot = ndimage.rotate(img, rot_angle, axes=[1,2], reshape=False)
    elif len(s) == 4:  # a set of 3D image
        img_rot = np.zeros(img.shape)
        for i in range(s[0]):
            img_rot[i] = ndimage.rotate(img[i], rot_angle, axes=[1,2], reshape=False)
    else:
        raise ValueError('Error! Input image has dimension > 4')
    img_rot[img_rot < 0] = 0

    return img_rot


def my_range(m_start, m_step, m_end):

    """
    Generate and iterate over a number sequence from m_start to m_end with step size of m_step

    Parameters:
    -----------

    m_start: int or float
    m_step: int or float
    m_end: int or float

    Return:
    -------

    Yield from the sequence

    """
    if m_step >= 0:
        while m_start <= m_end:
            yield m_start
            m_start += m_step
    if m_step < 0:
        while m_start >= m_end:
            yield m_start
            m_start += m_step


def re_projection(img3D, angle_list, ax=1):

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
            img_prj[i] = np.sum(rot3D(img3D, angle_list[i]), ax)

    elif ax == 2:  # sum up the colume --> projection from "left" to "right"
        img_prj = np.zeros([ang_size[0], im_size[0], im_size[1]])
        for i in range(ang_size[0]):
            print(f'current angle: {angle_list[j]}')
            img_prj[i] = np.sum(rot3D(img3D, angle_list[j]), ax)
    else:
        print('check "ax". Nothing done with the image!')
        return img3D
    return img_prj


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

#    leng = np.int(np.ceil(leng0 / np.cos(max(alfa,theta))))
    leng = leng0 + 30
    N1 = np.int16(np.ceil(leng * np.tan(alfa)))
    N2 = np.int16(np.ceil(leng * np.tan(theta)))

    Mask = np.zeros([2*N1-1, 2*N2-1, leng])

    s = Mask.shape
    s0= (2*N1_0-1, 2*N2_0-1, leng0) # size of "original" matrix

    M1 = np.zeros((s[0], s[2]))
    M11 = M1.copy()
    M2 = np.zeros((s[1], s[2]))
    M22 = M2.copy()

    for I in range(s[0]):
        for J in range(s[2]):
            i = I+1
            j = J+1
            if (np.abs(N1-i) >= j*np.tan(alfa)):
                M1[I,J] = 0
                M11[I,J] = M1[I,J]

            elif (np.abs(N1-i) < (j-1)*np.tan(alfa)
                        and (np.abs(N1-i)+1) > j*np.tan(alfa)):

                desi_1 = (j-1)*np.tan(alfa) - np.floor((j-1)*np.tan(alfa))
                desi_2 = j*np.tan(alfa) - np.floor(j*np.tan(alfa))
                M11[I,J] = 0.5 * (desi_1 + desi_2)
                M1[I,J] = M11[I,J] / np.cos(alfa);

            elif (np.abs(N1-i) < j*np.tan(alfa)
                        and (np.abs(N1-i)+1) > j*np.tan(alfa)
                        and np.abs(N1-i) > (j-1)*np.tan(alfa)):

                desi_1 = j*np.tan(alfa) - np.floor(j*np.tan(alfa))
                M11[I,J] = 0.5 * desi_1 * (desi_1/np.tan(alfa))
                M1[I,J] = M11[I,J] / np.cos(alfa)

            elif((np.abs(N1-i)+1) < j*np.tan(alfa)
                        and np.abs(N1-i) < (j-1)*np.tan(alfa)
                        and (np.abs(N1-i)+1) > (j-1)*np.tan(alfa)):
                desi_1 = np.ceil((j-1)*np.tan(alfa)) - (j-1)*np.tan(alfa)
                M11[I,J] = 1 - 0.5 * desi_1 * (desi_1/np.tan(alfa))
                M1[I,J] = M11[I,J] / np.cos(alfa)

            else:
                tmp = np.arctan(1.0*(N1-i)/j)
                M11[I,J] = 1
                M1[I,J] = M11[I,J] / np.cos(tmp)

    M1[N1-1,:] = 1
    M1[N1-1,0] = 0

    for K in range(s[1]):
        for J in range(s[2]):
            k = K+1
            j = J+1
            if np.abs(N2-k) >= j*np.tan(theta):
                M22[K,J]=0;
                M2[K,J]=M22[K,J]

            elif (np.abs(N2-k) < (j-1)*np.tan(theta)
                        and (np.abs(N2-k)+1) > j*np.tan(theta)):

                desk_1 = (j-1)*np.tan(theta) - np.floor((j-1)*np.tan(theta))
                desk_2 = j*np.tan(theta)-np.floor(j*np.tan(theta))
                M22[K,J] = 0.5 *(desk_1+desk_2)
                M2[K,J] = M22[K,J] / np.cos(theta)

            elif (np.abs(N2-k) < j*np.tan(theta)
                        and (np.abs(N2-k)+1) > j*np.tan(theta)
                        and np.abs(N2-k) > (j-1)*np.tan(theta)):

                desk_1=j*np.tan(theta)-np.floor(j*np.tan(theta))
                M22[K,J]=0.5*desk_1*(desk_1/np.tan(theta))
                M2[K,J]=M22[K,J]/np.cos(theta)

            elif((np.abs(N2-k)+1)<j*np.tan(theta)
                        and abs(N2-k)<(j-1)*np.tan(theta)
                        and (abs(N2-k)+1)>(j-1)*np.tan(theta)):

                desk_1=np.ceil((j-1)*np.tan(theta))-(j-1)*np.tan(theta)
                M22[K,J]=(1-0.5*desk_1*(desk_1/np.tan(theta)))
                M2[K,J]=M22[K,J]/np.cos(theta)

            else:
                tmp=np.arctan(1.0*(N2-k)/j)
                M22[K,J]=1
                M2[K,J]=M22[K,J]/np.cos(tmp)
        M2[N2-1,:] = 1
    M2[N2-1,0] = 0

    Mask1 = Mask.copy()
    Mask2 = Mask.copy()

    for i in range(s[1]):
        Mask1[:,i,:] = M1
    for i in range(s[0]):
        Mask2[i,:,:] = M2

    Mask = Mask1 * Mask2 # element by element multiply
    shape_mask = Mask > 0

    a,b,c = np.mgrid[1:s[0]+1, 1:s[1]+1, 1:s[2]+1]
    dis = np.sqrt((a-N1)**2 + (b-N2)**2 + (c-1)**2)
    '''
    dis = np.zeros(Mask.shape)

    for I in prange(s[0]):
        for J in prange(s[1]):
            for K in prange(s[2]):
                i=I+1;
                j=J+1;
                k=K+1;
                dis[I,J,K]=np.sqrt(((i-N1)**2+(j-N2)**2+(k-1)**2)*1.0)
    '''

    dis[N1-1,N2-1,0]=1
    dis = dis * shape_mask * 1.0
    a = np.floor(dis)  # integer part of "dis"
    b = dis - a   # decimal part of "dis"
    M_normal = np.zeros(Mask.shape)*shape_mask

    '''
    for i in range(1, np.int(np.max(dis))+1):
        flag_mask = np.ones(Mask.shape) # mark the position with radial distance == i
        temp = np.zeros(Mask.shape)

        b_mask1 = np.floor(dis) == i
        temp += shape_mask * (1-b) * b_mask1

        b_mask2 = np.floor(dis) == i-1
        temp += shape_mask * b * b_mask2

        b_mask3 = flag_mask > 0
        temp_sum = np.sum(temp * b_mask3)
        temp = temp / temp_sum

        flag_mask *= b_mask1 * b_mask2
        M_normal = M_normal + temp

    '''
    for i in range(1, np.int16(np.max(dis))+1):
        flag_mask = np.zeros(Mask.shape) # mark the position with radial distance == i
        temp = np.zeros(Mask.shape)

        ix,iy,iz = np.where(np.floor(dis) == i)

        if ix.size > 0:
            temp[ix,iy,iz] = temp[ix,iy,iz] + shape_mask[ix,iy,iz]*(1-b[ix,iy,iz])
            flag_mask[ix,iy,iz]=1

        ix,iy,iz = np.where(np.floor(dis) == i-1)
        if ix.size > 0:
            temp[ix,iy,iz] = temp[ix,iy,iz] + shape_mask[ix,iy,iz]*b[ix,iy,iz]
            flag_mask[ix,iy,iz]=1

#       x = np.unique(np.concatenate((x1, x2),axis=0)
#       ix,iy,iz=np.unravel_index(x, Mask.shape)
        ix,iy,iz = np.where(flag_mask > 0)
        temp[ix,iy,iz] = 1.0 * temp[ix,iy,iz] / np.sum(temp[ix,iy,iz])
        M_normal = M_normal + temp

    Mask3D = M_normal * Mask;

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

    ys = col - np.int(np.floor(s[2] / 2))
    ye = ys + s[2]
    if ys < 0: ys = 0
    if ye > sd[2]: ye = sd[2]+1

    zs = sli - np.int(np.floor(s[0] / 2))
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


def align_img(img_ref, img):

    """
    align images using dftreginstration (in align_class.py)

    Parameters:
    -----------

    img_ref: 2D array
          reference image

    img: 2D array
          image needs to align. Note: size(img) >= size(img_ref)

    Returns:
    --------

    img_align: 2D array
        aligned image with same size of img_ref

    row_shift: float
        shift amount in row

    col_shift: float
        shift amount in col

    """

    img_ref = np.array(img_ref)
    img = np.array(img)
    assert(len(img_ref.shape) == 2), "reference image should be 2D image"
    assert(len(img.shape) == 2), "image need to align should be 2D image"

    row_ref, col_ref = img_ref.shape
    ref = np.zeros(img.shape)
    ref[0:row_ref, 0:col_ref] = img_ref
    ref_fft = np.fft.fft2(ref)
    img_fft = np.fft.fft2(img)
    error, diffphase, row_shift, col_shift, image_reg = dftregistration(ref_fft,img_fft,100)
    temp = ndimage.interpolation.shift(img, [row_shift, col_shift])

    img_align = temp[0:row_ref, 0:col_ref]

    return img_align, row_shift, col_shift



def write_attenuation(elem, data, current_angle, file_path='./Angle_prj'):

    """
    write attenuation coefficient into file in the directory of './Angle_prj' as:

    'atten_gd_prj_-45.h5'

    where:  elem = 'gd'
            current_angle = -45

    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not(os.path.isdir(file_path)):
        os.mkdir(file_path)
    os.chdir(file_path)
    fname = 'atten_' + elem + '_prj_' + f'{current_angle:04d}' + '.tiff'
    io.imsave(fname, data.astype(np.float32))
    '''
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('dataset_1', data=data)
    '''
    os.chdir(dir_path)


def write_projection(mode, elem, data, current_angle, file_path='./Angle_prj'):

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

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not(os.path.isdir(file_path)):
        os.mkdir(file_path)
    os.chdir(file_path)
    if mode == 'single_file' or mode == 's':
        fname = elem + '_ref_prj_single_file.h5'
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('dataset_1', data=data)
            hf.create_dataset('angle_list', data=current_angle)
    elif mode == 'multi_file' or mode == 'm':
        fname = elem + '_ref_prj_' + f'{current_angle:04d}' + '.tiff'
        print(fname)
        io.imsave(fname, data.astype(np.float32))
        '''
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('dataset_1', data=data)
        '''
    else: print ('unrecongnized "mode"')
    os.chdir(dir_path)


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

def get_atten_coef(elem_type, XEng, em_E):
    '''
    elem_type = ['Zr', 'La', 'Hf']
    XEng = 12
    em_E = [4.6, ]
    '''
    et = elem_type
    cs = {}
    n = len(elem_type)
    cs['elem_type'] = elem_type
    for i in range(n):
        # atten at incident x-ray
        cs[f'{et[i]}-x'] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(et[i]), XEng)
        for j in range(n):
            # atten at each emission line
            cs[f'{et[i]}-{et[j]}'] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(et[i]), em_E[et[j]])
    return cs

def cal_atten_with_direction0(img4D, cs, param, Mask3D, position_det='r'):
    """
    Calculate the attenuation of incident x-ray, fluorescent with given
    experiment configuration. Assume x-ray is passing from front to the back
    of the 3D object

    Will change configuration to:
    x-ray: pass from top->r_down (go through slices)
    detector: sit at bottom of each slice of 3D image

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
    global mask3D

    mask3D = Mask3D

    elem_type = cs['elem_type']
    n_type = img4D.shape[0]

    # change x-ray passing from top -> down (go through slices)
    img4D_t = np.transpose(img4D, [0, 2, 1, 3])
    img4D_t = img4D_t[:, ::-1]
    # change detector position at the bottom of the slice image
    angle = -90 if position_det=='r' else 90
    img4D_r = rot3D(img4D_t, angle)
    atten, atten_fl, atten_xray = cal_atten_3D0(img4D_r, cs, param, display_flag=False)

    atten3D = {}
    angle = 90 if position_det=='r' else -90
    for i in range(n_type):
        ele = elem_type[i]
        atten3D[ele] = rot3D(atten[ele], angle)
        atten3D[ele] = atten3D[ele][::-1]
        atten3D[ele] = np.transpose(atten3D[ele], [1,0,2])
    return atten3D



def cal_atten_with_direction(img4D, cs, param, Mask3D, position_det='r', num_cpu=8):
    """
    Calculate the attenuation of incident x-ray, fluorescent with given
    experiment configuration. Assume x-ray is passing from front to the back
    of the 3D object

    Will change configuration to:
    x-ray: pass from top->r_down (go through slices)
    detector: sit at bottom of each slice of 3D image

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
    global mask3D

    mask3D = Mask3D


    elem_type = cs['elem_type']
    n_type = img4D.shape[0]

    # change x-ray passing from top -> down (go through slices)
    img4D_t = np.transpose(img4D, [0, 2, 1, 3])
    img4D_t = img4D_t[:, ::-1]
    # change detector position at the bottom of the slice image
    angle = -90 if position_det=='r' else 90
    img4D_r = rot3D(img4D_t, angle)
    atten, atten_fl, atten_xray = cal_atten_3D(img4D_r, cs, param, mask3D, display_flag=False, num_cpu=num_cpu)

    atten3D = {}
    angle = 90 if position_det=='r' else -90
    for i in range(n_type):
        ele = elem_type[i]
        atten3D[ele] = rot3D(atten[ele], angle)
        atten3D[ele] = atten3D[ele][::-1]
        atten3D[ele] = np.transpose(atten3D[ele], [1,0,2])
    return atten3D


def rm_nan(*args):

    """
    Remove nan and inf in data
    e.g. a =  rm_nan(data1, data2, data3)

    Parameters:
    -----------

    args: a list of ndarray data with same shape

    Return:
    -------

    ndarray

    """

    num = len(args)
    s = args[0].shape

    data = np.zeros([num] + list(s))
    for i in range(num):
        data[i] = args[i]
    data = np.array(data)
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    data = np.squeeze(data)

    return np.array(data)


def im_bin(img, binning=2, mode='xyz'):

    """
    Image binning

    Parameters:
    -----------
    img: 2D or 3D array

    binning:  int

    mode: char (for 3d image only)
          mode ='xyz' --> binning all direction
          mode ='xy'  --> no binning on ax=0

    Returns:
    --------
    Binned image

    """

    img = rm_nan(img)

    s = np.array(img.shape, dtype=int)
    dim = s % binning
    n_copy = int(binning)

    if len(s) == 2:
        s1 = s - dim
        temp = img[0:s1[0], 0:s1[1]]
        data = np.zeros([n_copy, s1[0]/n_copy, s1[1]/n_copy])
        for i in range(n_copy):
            data[i] = data[i] + temp[i::n_copy,i::n_copy]
        img1 = sum(data, 0)/float(n_copy)


    if len(s) == 3:
        if mode == 'xyz':
            s1 = s - dim
            temp = img[0:s1[0], 0:s1[1], 0:s1[2]]
            data = np.zeros([n_copy, s1[0]/n_copy, s1[1]/n_copy, s1[2]/n_copy])
            for i in range(n_copy):
                data[i] = data[i] + temp[i::n_copy,i::n_copy, i::n_copy]
            img1 = sum(data, 0)/float(n_copy)
        if mode == 'xy':
            s1 = s - dim
            temp = img[0:s[0], 0:s1[1], 0:s1[2]]
            data = np.zeros([n_copy, s[0], s1[1]/n_copy, s1[2]/n_copy])
            for i in range(n_copy):
                data[i] = data[i] + temp[:, i::n_copy, i::n_copy]
            img1 = sum(data, 0)/float(n_copy)


    return img1


def generate_H(elem_type, ref3D_tomo, sli, angle_list, bad_angle_index=[], file_path='./Angle_prj', flag=1):

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

    # fpre_att = file_path + '/atten_' + elem_type.lower() + '_prj_'
    fpre_att = file_path + '/atten_' + elem_type + '_prj_'

    ref_tomo = ref3D_tomo.copy()
    theta = np.array(angle_list / 180 * np.pi)
    num = len(theta) - len(bad_angle_index)
    s = ref_tomo.shape
    cx = (s[2]-1) / 2.0       # center of col
    cy = (s[1]-1) / 2.0       # center of row

    '''
    if flag:
        print('Reading attenuation from file ...')
    else:
        print('No attenation read, generate non-attenuated matrix ...')
    '''
    H_tot = np.zeros([s[2]*num, s[2]*s[2]])
    k = -1
    for i in range(len(theta)):
#        print(f'current theta {i}')
        if i in bad_angle_index:
            continue
        k = k + 1
#        print(f'current angle: {angle_list[i]}')
        if flag:
            f_att = fpre_att + f'{angle_list[i]:04d}.tiff'
            att = io.imread(f_att)[sli]
            '''
            f = h5py.File(f_att,'r')
            att = np.array(f['dataset_1'][sli])
            f.close()
            '''
        else:
            att = np.ones([s[1],s[2]])

        T = np.array([[np.cos(-theta[i]), -np.sin(-theta[i])],[np.sin(-theta[i]), np.cos(-theta[i])]])
        H = np.zeros([s[2], s[2]*s[2]])
        for col in range(s[2]):
            for row in range(s[1]):
                p = row
                q = col
                cord = np.dot(T,[[p-cx],[q-cy]]) + [[cx],[cy]]
                if ((cord[0] > s[1]-1) or (cord[0] <= 0) or (cord[1] > s[2]-1) or (cord[1] <= 0)):    continue
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
        H_tot[k*s[2] : (k+1)*s[2], :] = H

    return H_tot


def generate_I(elem_type, ref3D_tomo, sli, angle_list, bad_angle_index=[], file_path='./Angle_prj'):

    """
    Generate matriz I for solving eqution: H*C=I
    In folder of file_path:
            Needs aligned 2D projection at each rotation angle
    e.g. I = generate_I(Gd, 30, angle_list, bad_angle_index, file_path='./Angle_prj')

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
         aligned projection image:         e.g. Gd_ref_prj_50.0.h5
         these files can be generated through function of: 'write_projection'

    Returns:
    --------
    1D array

    """

    #print('Reding aligned projection and generate matrix "I"...')

    # fpre_prj = file_path + '/' + elem_type.lower() + '_ref_prj_'
    fpre_prj = file_path + '/' + elem_type + '_ref_prj_'
    theta = np.array(angle_list / 180 * np.pi)
    num = len(theta) - len(bad_angle_index)
    s = ref3D_tomo.shape
    I_tot = np.zeros(s[2]*num)
    k = -1
    for i in range(len(theta)):
        if i in bad_angle_index:
            continue
        k = k + 1
        f_ref = fpre_prj + f'{angle_list[i]:04d}.tiff'
        '''
        try:
            f = h5py.File(f_ref, 'r')
        except:
            print(f'file "{f_ref}" does not exist ...')
            return 0
        prj = np.array(f['dataset_1'][sli])
        f.close()
        '''
        prj = io.imread(f_ref)[sli]
        I_tot[k*s[2] : (k+1)*s[2]] = prj

    return I_tot


def load_param(fn):

    """
    Read parameter files with format of:

    X-ray Energy:	  12
    Number of elements:	   3
    Element type: Zr, La, Hf
    Pixel size(nm):    50
    density(g/cm3):    5.11
    emission line energy: 2.044, 4.647, 7.899
    emission cross section(cm2/g): 1.313, 3.421, 14.97

    """

    with open(fn) as f:
        lines = f.readlines()

    lines = [x.strip('\n') for x in lines]
    lines = [x.replace('\t', ' ') for x in lines]
    lines = [x for x in lines if x]
    lines = [x.strip(' ') for x in lines]

    for i in range(np.size(lines)):
        if lines[i].find('#') >-1:
            continue
        words = lines[i].split(':')
        if words[0].lower().find('x-ray energy')>-1:
            XEng = float(words[1].strip(' '))
            print(f'XEng = {XEng} keV')

        elif words[0].lower().find('number')>-1:
            nelem = int(words[1].strip(' ' ))
            print(f'num of elem = {nelem}')

        elif words[0].lower().find('mass') > -1:
            mass = [float(x.strip(' ')) for x in words[1].split(',')]
            print(f'mass = {mass} g/mol')

        elif words[0].lower().find('thickness') > -1:
            thick = float(words[1].strip(' '))
            print(f'max thickness = {thick} um')

        elif words[0].lower().find('pixel')>-1:
            pix = float(words[1].strip(' '))
            pix = pix * 1e-7 # unit of cm
            print(f'pix size = {pix:3.1e} cm')

        elif words[0].lower().find('density')>-1:
            rho = float(words[1].strip(' '))
            print(f'rho = {rho} g/cm3')

        elif words[0].lower().find('emission energy')>-1:
            em_eng = [float(x.strip(' ')) for x in words[1].split(',')]
            print(f'em_eng = {em_eng} keV')

        elif words[0].lower().find('cross section')>-1:
            cs_em = [float(x.strip(' ')) for x in words[1].split(',')]
            print(f'cs_em = {cs_em} cm2/g')

        elif words[0].lower().find('element type')>-1:
            elem_type = [x.strip(' ') for x in words[1].split(',')]
            print(f'element type: {elem_type}')

    M = {} # mole mass
    em_E = {} # emission energy (keV)
    em_cs = {} # emission cross section (cm2/g)
    for i in range(nelem):
        M[elem_type[i]] = mass[i]
        em_E[elem_type[i]] = em_eng[i]
        em_cs[elem_type[i]] = cs_em[i]
    res = {}
    res['XEng'] = XEng
    res['nelem'] = nelem
    res['rho'] = rho
    res['pix'] = float(f'{pix:3.1e}')
    res['M'] = M
    res['em_E'] = em_E
    res['em_cs'] = em_cs
    res['elem_type'] = elem_type
    res['img_thick'] = int(thick*1e-4 / pix)
    return res


def atten_2D_slice(sli, mu_ele, display_flag=True):
    global mask3D
    # calculate pixel attenuation length
    if display_flag:
        print(f'sli={sli}')
    s_mu = mu_ele.shape
    atten_ele = np.ones([s_mu[1], s_mu[2]])
    #sli = int(s_mu[0]/2)
    for i in np.arange(s_mu[1]): # row
        length = max(s_mu[1] - i, 7)
        mask = np.asarray(mask3D[f'{length}'], dtype='f4')
        for j in np.arange(0, s_mu[2]): # column
            #if mu_ele[sli, i, j] == 0:
            #    continue
            atten_ele[i, j] = retrieve_data_mask(mu_ele, int(i), int(j), sli, mask)
    return atten_ele


def atten_2D_slice_mpi(sli, mu_ele, display_flag=True, num_cpu=4):

    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    #from tqdm.contrib.concurrent import process_map
    from functools import partial
    max_num_cpu = round(cpu_count() * 0.8)
    if num_cpu == 0 or num_cpu > max_num_cpu:
        num_cpu = max_num_cpu
    print(f'assembling slice using {num_cpu:2d} CPUs')
    partial_func = partial(atten_2D_slice, mu_ele=mu_ele, display_flag=display_flag)
    pool = Pool(num_cpu)
    #res = pool.map(partial_func, sli)
    res = []
    for result in tqdm(pool.imap(func=partial_func, iterable=sli), total=len(sli)):
        res.append(result)
    pool.close()
    pool.join()
    #res = process_map(atten_2D_slice, sli, mu_ele, False, max_worder=num_cpu)
    return res


def atten_voxel(idx, row, mu_ele):
    global mask3D
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



def rm_abnormal(img):
    t = img.copy()
    t[np.isnan(t)] = 0
    t[np.isinf(t)] = 0
    return t


def cal_frac(*args):
    if len(args) == 1:
        img = args[0].copy()
        s = img.shape
        if len(s) == 2:
            img = np.expand_dims(img, 0)
    else:
        img = pre_treat(*args)

    img_sum = 0
    s = img.shape
    n = s[0]
    frac = np.zeros(s)
    img_sum = np.sum(img, axis=0)
    img_sum = scipy.ndimage.median_filter(img_sum, 3)
    img_sum = rm_abnormal(img_sum)

    sum_flat = img_sum.flatten()
    sum_sort = np.sort(sum_flat)

    # pix_max = np.max(img_sum)
    pix_max = sum_sort[int(0.95*len(sum_sort))]
    img_mask = img_sum > (pix_max)*1e-4
    img_sum[~img_mask] = 0
    for i in range(n):
        tmp = rm_abnormal(img[i] / img_sum)
        tmp[tmp>1] = 1
        frac[i] = tmp
    res = {}
    res['img_sum'] = img_sum
    res['frac'] = frac
    res['pix_max'] = pix_max
    res['img_mask'] = img_mask
    return res


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
    s0 =np.int16(data3d.shape)
    s = np.int16(mask.shape)
    xs = np.int(row)
    xe = np.int(row + s[1])
    ys = np.int(col - np.floor(s[2]/2))
    ye = np.int(col + np.floor(s[2]/2)+1)
    zs = np.int(sli - np.floor(s[0]/2))
    ze = np.int(sli + np.floor(s[0]/2)+1)
    ms = mask.shape
    m_xs = 0;   m_xe = ms[1];
    m_ys = 0;   m_ye = ms[2];
    m_zs = 0;   m_ze = ms[0];
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


def cal_atten_3D0(img4D, cs, param, display_flag=True):
    from tqdm import tqdm
    elem_type = cs['elem_type']
    rho = param['rho']
    pix = param['pix']
    img_thick = param['img_thick']
    res = cal_frac(img4D)
    n_type = img4D.shape[0]
    img_sum = res['img_sum']
    pix_max = res['pix_max']
    frac = res['frac']
    s = img_sum.shape # (200, 200,200), detector is at bottom of image
    cs_mix = {} # effective cross section at each fluorescent line
    mu = {}     # atten coef: cm2/g * g/cm3 -> cm-1
    mu3D = {}   # atten coef of 3D
    atten_single_pix = {} # single-pix atten coef of each element
    atten_xray = {}
    atten = {}
    for i in range(n_type):
        ele = elem_type[i]
        cs_mix[ele] = 0
        for j in range(n_type):
            ele2 = elem_type[j]
            cs_mix[ele] += frac[j] * cs[f'{ele2}-{ele}']
        cs_mix[ele] *= (img_sum/pix_max)
    for i in range(n_type):
        ele = elem_type[i]
        mu[ele] = cs_mix[ele] * rho
    for i in range(n_type):
        ele = elem_type[i]
        print(f'calculating attenuation for {ele}')
        ts = time.time()
        atten_tmp = np.ones(s)
        for j in np.arange(s[0]):
            sli = j
            atten_tmp[j] = atten_2D_slice(sli, mu[ele], display_flag=display_flag)
        atten_single_pix[ele] = np.exp(-atten_tmp * pix)
        te = time.time()
        print(f'taking {te-ts:3.1f} sec')
    ### including atten_incident x-ray ###
    #print('calculating x-ray attenuation')
    cs_mix['x'] = 0
    for i in range(n_type):
        ele = elem_type[i]
        cs_mix['x'] += frac[i] * cs[f'{ele}-x'] * img_sum/pix_max
    mu['x'] = cs_mix['x'] * rho
    x_ray_atten = np.ones(s)
    for j in range(1, s[0]):
        x_ray_atten[j] = x_ray_atten[j-1] * np.exp(-mu['x'][j] * pix)
    for i in range(n_type):
        ele = elem_type[i]
        atten[ele] = atten_single_pix[ele] * x_ray_atten
    return atten, atten_single_pix, x_ray_atten

def cal_atten_3D_row_by_row(img4D, cs, param, display_flag=True, num_cpu=8):
    global mask3D
    elem_type = cs['elem_type']
    rho = param['rho']
    pix = param['pix']
    img_thick = param['img_thick']
    res = cal_frac(img4D)
    n_type = img4D.shape[0]
    img_sum = res['img_sum']
    pix_max = res['pix_max']
    frac = res['frac']
    s = img_sum.shape # (200, 200,200), detector is at bottom of image
    cs_mix = {} # effective cross section at each fluorescent line
    mu = {}     # atten coef: cm2/g * g/cm3 -> cm-1
    mu3D = {}   # atten coef of 3D
    atten_single_pix = {} # single-pix atten coef of each element
    atten_xray = {}
    atten = {}
    for i in range(n_type):
        ele = elem_type[i]
        cs_mix[ele] = 0
        for j in range(n_type):
            ele2 = elem_type[j]
            cs_mix[ele] += frac[j] * cs[f'{ele2}-{ele}']
        cs_mix[ele] *= (img_sum/pix_max)
    for i in range(n_type):
        ele = elem_type[i]
        mu[ele] = cs_mix[ele] * rho
    for i in range(n_type):
        ele = elem_type[i]
        print(f'calculating attenuation for {ele}')
        ts = time.time()

        atten_tmp = np.ones(s)

        for row in trange(s[1]):
            atten_tmp[:, row] = atten_2D_row_mpi(row, mu[ele], num_cpu=num_cpu)
        atten_tmp = np.array(atten_tmp)
        atten_single_pix[ele] = np.exp(-atten_tmp * pix)
        te = time.time()
        print(f'taking {te-ts:3.1f} sec')
    ### including atten_incident x-ray ###
    #print('calculating x-ray attenuation')
    cs_mix['x'] = 0
    for i in range(n_type):
        ele = elem_type[i]
        cs_mix['x'] += frac[i] * cs[f'{ele}-x'] * img_sum/pix_max
    mu['x'] = cs_mix['x'] * rho
    x_ray_atten = np.ones(s)
    for j in range(1, s[0]):
        x_ray_atten[j] = x_ray_atten[j-1] * np.exp(-mu['x'][j] * pix)
    for i in range(n_type):
        ele = elem_type[i]
        atten[ele] = atten_single_pix[ele] * x_ray_atten
    return atten, atten_single_pix, x_ray_atten


def cal_atten_3D(img4D, cs, param, Mask3D, display_flag=True, num_cpu=8):
    global mask3D
    mask3D = Mask3D
    elem_type = cs['elem_type']
    rho = param['rho']
    pix = param['pix']
    img_thick = param['img_thick']
    res = cal_frac(img4D)
    n_type = img4D.shape[0]
    img_sum = res['img_sum']
    pix_max = res['pix_max']
    frac = res['frac']
    s = img_sum.shape # (200, 200,200), detector is at bottom of image
    cs_mix = {} # effective cross section at each fluorescent line
    mu = {}     # atten coef: cm2/g * g/cm3 -> cm-1
    mu3D = {}   # atten coef of 3D
    atten_single_pix = {} # single-pix atten coef of each element
    atten_xray = {}
    atten = {}
    for i in range(n_type):
        ele = elem_type[i]
        cs_mix[ele] = 0
        for j in range(n_type):
            ele2 = elem_type[j]
            cs_mix[ele] += frac[j] * cs[f'{ele2}-{ele}']
        cs_mix[ele] *= (img_sum/pix_max)
    for i in range(n_type):
        ele = elem_type[i]
        mu[ele] = cs_mix[ele] * rho
    for i in range(n_type):
        ele = elem_type[i]
        print(f'calculating attenuation for {ele}')
        ts = time.time()
        sli = np.arange(s[0])
        atten_tmp = atten_2D_slice_mpi(sli, mu[ele], num_cpu=num_cpu, display_flag=display_flag)
        atten_tmp = np.array(atten_tmp)
        atten_single_pix[ele] = np.exp(-atten_tmp * pix)
        te = time.time()
        print(f'taking {te-ts:3.1f} sec')
    ### including atten_incident x-ray ###
    #print('calculating x-ray attenuation')
    cs_mix['x'] = 0
    for i in range(n_type):
        ele = elem_type[i]
        cs_mix['x'] += frac[i] * cs[f'{ele}-x'] * img_sum/pix_max
    mu['x'] = cs_mix['x'] * rho
    x_ray_atten = np.ones(s)
    for j in range(1, s[0]):
        x_ray_atten[j] = x_ray_atten[j-1] * np.exp(-mu['x'][j] * pix)
    for i in range(n_type):
        ele = elem_type[i]
        atten[ele] = atten_single_pix[ele] * x_ray_atten
    return atten, atten_single_pix, x_ray_atten

@njit
def generate_H_jit(elem_type, ref3D_tomo, sli, angle_list, att_assemble, H_tot, H_zero):

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

    #theta = np.array(angle_list / 180 * np.pi)
    num = len(theta)
    s = ref3D_tomo.shape
    cx = (s[2]-1) / 2.0       # center of col
    cy = (s[1]-1) / 2.0       # center of row
    #H_tot = np.zeros([s[2]*num, s[2]*s[2]])
    for i in prange(len(theta)):
        att = att_assemble[i]

        T = np.array([[np.cos(-theta[i]), -np.sin(-theta[i])],[np.sin(-theta[i]), np.cos(-theta[i])]])
        H = H_zero.copy()
        for col in range(s[2]):
            for row in range(s[1]):
                p = row
                q = col
                cord = np.dot(T,[[p-cx],[q-cy]]) + [[cx],[cy]]
                if ((cord[0] > s[1]-1) or (cord[0] <= 0) or (cord[1] > s[2]-1) or (cord[1] <= 0)):    continue
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

def bin_ndarray(ndarray, new_shape=None, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if new_shape == None:
        s = np.array(ndarray.shape)
        s1 = np.int32(s/2)
        new_shape = tuple(s1)
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



    #
