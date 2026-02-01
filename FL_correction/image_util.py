import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_yen, threshold_multiotsu
from skimage.filters import gaussian as gf
from multiprocessing import Pool, cpu_count
from pystackreg import StackReg
from tqdm import tqdm, trange
from functools import partial
import bm3d

def circle_mask(img3D, ratio=1, val=0):
    s = img3D.shape
    if len(s) == 2:
        img3D = img3D[np.newaxis]
    s = img3D.shape
    im = np.zeros_like(img3D)
    x = np.arange(s[0])
    y = np.arange(s[2])
    X, Y = np.meshgrid(y, x)
    X = X / s[2]
    Y = Y / s[1]
    mask = np.float32(((X-0.5)**2 + (Y-0.5)**2)<(ratio/2)**2)
    mask_minus = 1 - mask
    for i in range(s[0]):
        im[i] = img3D[i] * mask + (mask_minus) * val
    return im



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


def fast_rot90_3D(img_raw, ax=0, mode='c-clock'):
    '''
    ax: rotation axes
    ax=0: positive direction from bottom --> top
    ax=1: positive direction from front --> back (this is un-conventional to righ-hand-rule)
    ax=2: positive direction from left --> right 
        
    mode: 
        'clock': rotate clockwise --> "-90 degree"
        'c-clock': rotate count-clockwise --> "90 degree"
    '''
    img_r = img_raw.copy()
    if ax == 0:
        img_r = np.transpose(img_r, [0, 2, 1])
        if mode == 'clock':
            img_r = img_r[:, :, ::-1]
        elif mode == 'c-clock':
            img_r = img_r[:, ::-1, :]
    if ax == 1:
        img_r = np.transpose(img_r, [2, 1, 0])
        if mode == 'clock':
            img_r = img_r[:, :, ::-1]
        elif mode == 'c-clock':
            img_r = img_r[::-1, :, :]
    if ax == 2:
        img_r = np.transpose(img_r, [1, 0, 2])
        if mode == 'clock':
            img_r = img_r[::-1, :, :]
        elif mode == 'c-clock':
            img_r = img_r[:, ::-1, :]
    return img_r      

def fast_rot90_4D(img4D, ax=0, mode='c-clock'):
    s = img4D.shape
    t = fast_rot90_3D(img4D[0], ax, mode)
    ss = t.shape
    img4D_r = np.zeros([s[0], ss[0], ss[1], ss[2]])
    img4D_r[0] = t
    for i in trange(1, s[0]):
        img4D_r[i] = fast_rot90_3D(img4D[i], ax, mode)
    return img4D_r


def rot3D_dict_img(img_dict, rot_angle):
    dict_rot = img_dict.copy()
    keys = img_dict.keys()
    for k in keys:
        img = img_dict[k]
        img_r = rot3D(img, rot_angle)
        dict_rot[k] = img_r.copy()
    return dict_rot


def rot3D(img_raw, rot_angle, order=1, reshape=False):

    """
    Rotate 2D or 3D or 4D(set of 3D) image with angle = rot_angle
    rotate anticlockwise

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

    img = np.array(img_raw, dtype=np.float32)
    img = rm_nan(img)
    if rot_angle == 0:
        return img
    s = img.shape
    if np.mod(rot_angle, 90) == 0:
        order = 0
    if len(s) == 2:    # 2D image
        img_rot = ndimage.rotate(img, rot_angle, order=order, reshape=reshape)
    elif len(s) == 3:  # 3D image, rotating along axes=0
        img_rot = ndimage.rotate(img, rot_angle, axes=[1,2], order=order, reshape=reshape)
    elif len(s) == 4:  # a set of 3D image
        img_rot = []
        for i in trange(s[0]):
            new_img = ndimage.rotate(img[i], rot_angle, axes=[1,2], order=order, reshape=reshape)
            img_rot.append(new_img)
        img_rot = np.array(img_rot)
    else:
        raise ValueError('Error! Input image has dimension > 4')
    img_rot[img_rot < 0] = 0

    return img_rot


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


def align_img(img_ref, img, align_flag=1, method='translation'):
    '''
    :param img_ref: reference image
    :param img: image need to align
    :param align_flag: 1: will do alignment; 0: output shift list only
    :param method:
        'translation': x, y shift
        'rigid': translation + rotation
        'scaled rotation': translation + rotation + scaling
        'affine': translation + rotation + scaling + shearing
    :return:
        align_flag == 1: img_ali, row_shift, col_shift, sr (row_shift and col_shift only valid for translation)
        align_flag == 0: row_shift, col_shift, sr (row_shift and col_shift only valid for translation)
    '''
    assert(len(img_ref.shape) == 2), "reference image should be 2D image"
    assert(len(img.shape) == 2), "image need to align should be 2D image"
    if method == 'translation':
        sr = StackReg(StackReg.TRANSLATION)
    elif method == 'rigid':
        sr = StackReg(StackReg.RIGID_BODY)
    elif method == 'scaled rotation':
        sr = StackReg(StackReg.SCALED_ROTATION)
    elif method == 'affine':
        sr = StackReg(StackReg.AFFINE)
    else:
        sr = [[1, 0, 0],[0, 1, 0], [0, 0, 1]]
        print('unrecognized align method, no aligning performed')
    tmat = sr.register(img_ref, img)
    row_shift = -tmat[1, 2]
    col_shift = -tmat[0, 2]
    if align_flag:
        img_ali = sr.transform(img)
        return img_ali, row_shift, col_shift
    else:
        return row_shift, col_shift, sr

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

def adaptive_threshold(img, fill_hole=False, dilation=0, erosion=0):
    s = img.shape
    mask = np.ones(s)
    struct = ndimage.generate_binary_structure(2, 1)
    if len(s) == 3:
        for i in range(s[0]):
            image = img[i]
            try:
                thresholds = threshold_multiotsu(image)
                regions = np.digitize(image, bins=thresholds)
                mask[i][regions<1] = 0
                if fill_hole:
                    mask[i] = ndimage.binary_fill_holes(mask[i], np.ones((5, 5)))
                if dilation > 0:
                    mask[i] = ndimage.binary_dilation(mask[i], structure=struct, iterations=dilation)
                if erosion > 0:
                    mask[i] = ndimage.binary_erosion(mask[i], structure=struct, iterations=erosion)
            except Exception as err:
                mask[i] = 0

    elif len(s) == 2:
        image = img
        try:
            thresholds = threshold_multiotsu(image)
            regions = np.digitize(image, bins=thresholds)
            mask[regions<1] = 0
            if fill_hole:
                 mask = ndimage.binary_fill_holes(mask, np.ones((5, 5)))
            if dilation > 0:
                mask = ndimage.binary_dilation(mask, structure=struct, iterations=dilation)
            if erosion > 0:
                mask = ndimage.binary_erosion(mask, structure=struct, iterations=erosion)

        except Exception as err:
            mask[:] = 0

    else:
        print('image shape not recognized')
    return mask


def rm_boarder(img_array, w=5):
    img = img_array.copy()
    s = img.shape
    if len(s) == 3:
        img[:, :w] = 0
        img[:, -w:] = 0
        img[:, :, :w] = 0
        img[:, :, -w:] = 0
    if len(s) == 4:
        img[:, :, :w] = 0
        img[:, :, -w:] = 0
        img[:, :, :, :w] = 0
        img[:, :, :, -w:] = 0
    return img 
############################

def rm_abnormal(img):
    tmp = img.copy()
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    tmp[tmp < 0] = 0
    return tmp


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
    data = data[0]

    return np.array(data)



def img_smooth(img, kernal_size, axis=0):
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()

    if axis == 0:
        for i in range(img_stack.shape[0]):
            img_stack[i] = medfilt2d(img_stack[i], kernal_size)
    elif axis == 1:
        for i in range(img_stack.shape[1]):
            img_stack[:, i] = medfilt2d(img_stack[:,i], kernal_size)
    elif axis == 2:
        for i in range(img_stack.shape[2]):
            img_stack[:, :, i] = medfilt2d(img_stack[:,:, i], kernal_size)
    return img_stack


def otsu_mask(img, kernal_size, iters=1, bins=256, erosion_iter=0):
    img_s = img.copy()
    img_s[np.isnan(img_s)] = 0
    img_s[np.isinf(img_s)] = 0
    for i in range(iters):
        img_s = img_smooth(img_s, kernal_size)
    thresh = threshold_otsu(img_s, nbins=bins)
    mask = np.zeros(img_s.shape)
    #mask = np.float32(img_s > thresh)
    mask[img_s > thresh] = 1
    mask = np.squeeze(mask)
    if erosion_iter:
        struct = ndimage.generate_binary_structure(2, 1)
        struct1 = ndimage.iterate_structure(struct, 2).astype(int)
        mask = ndimage.binary_erosion(mask, structure=struct1).astype(mask.dtype)
    mask[:erosion_iter+1] = 1
    mask[-erosion_iter-1:] = 1
    mask[:, :erosion_iter+1] = 1
    mask[:, -erosion_iter-1:] = 1
    return mask

def otsu_mask_stack(img, kernal_size, iters=1, bins=256, erosion_iter=0):
    s = img.shape
    img_m = np.zeros(s)
    for i in trange(s[0]):
        img_m[i] = otsu_mask(img[i], kernal_size, iters, bins, erosion_iter)
    img_r = img * img_m
    return img_r
        


def rm_noise(img, noise_level=2e-3, filter_size=3):
    img_s = medfilt2d(img, filter_size)
    id0 = img_s==0
    img_s[id0] = img[id0]
    img_diff = np.abs(img - img_s)
    index = img_diff > noise_level
    img_m = img.copy()
    img_m[index] = img_s[index]
    return img_m


def rm_noise2(img, noise_level=0.02, filter_size=3):
    img_s = medfilt2d(img, filter_size)
    id0 = img==0
    img_s[id0] = img[id0]
    img_diff = np.abs((img - img_s) / img)
    index = img_diff > noise_level
    img_m = img.copy()
    img_m[index] = img_s[index]
    return img_m


def rm_noise2_stack(img_stack, noise_level=0.02, filter_size=3):
    s = img_stack.shape
    img1 = img_stack.copy()
    if len(s) == 3: # 3D stack
        for i in range(s[0]):
            img = img_stack[i]
            img_s = medfilt2d(img, filter_size)
            id0 = img == 0
            img_s[id0] = img[id0]
            img_diff = (img - img_s) / img
            index = img_diff > noise_level
            img_m = img.copy()
            img_m[index] = img_s[index]
            img1[i] = img_m
    if len(s) == 4:
        for i in trange(s[0]):
            img1[i] = rm_noise2_stack(img_stack[i], noise_level, filter_size)
    return img1


def img_denoise_bm3d(img, sigma=0.01):
    try:
        import bm3d
        s = img.shape
        if len(s) == 2:
            img_stack = img.reshape(1, s[0], s[1])
        else:
            img_stack = img.copy()
        img_d = img_stack.copy()
        n = img_stack.shape[0]
        for i in range(n):
            img_d[i] = bm3d.bm3d(img_stack[i], sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        return img_d
    except Exception as err:
        print(err)
        return img

def img_denoise_bm3d_single(img, sigma=0.1):
    # img.shape =(100, 100)
    try:
        #import bm3d
        img_d = bm3d.bm3d(img, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        return img_d
    except Exception as err:
        print(err)
        return img


def img_denoise_bm3d_mpi(img, sigma=0.1, n_cpu=8):
    max_cpu = round(cpu_count() * 0.8)
    n_cpu = min(n_cpu, max_cpu)
    n_cpu = max(n_cpu, 1)
    s = img.shape
    pool = Pool(n_cpu)
    res = []
    partial_func = partial(img_denoise_bm3d_single, sigma=sigma)
    for result in tqdm(pool.imap(func=partial_func, iterable=img), total=len(img)):
        res.append(result)
    pool.close()
    pool.join()
    img_d = np.array(res)
    return img_d


def img_denoise_nl_single(img, patch_size=5, patch_distance=6):
    img_d = img.copy()
    patch_kw = dict(patch_size=patch_size,  # 5x5 patches
                    patch_distance=patch_distance,  # 13x13 search area
                    )
    sigma_est = np.mean(estimate_sigma(img_d))
    img_d = denoise_nl_means(img_d, h=1.2 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    return img_d


def img_denoise_nl(img, patch_size=5, patch_distance=6):
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()
    img_d = img_stack.copy()
    n = img_stack.shape[0]
    for i in range(n):
        img_d[i] = img_denoise_nl_single(img_stack[i], patch_size, patch_distance)
    return img_d


def img_denoise_nl_mpi(img, patch_size=5, patch_distance=6, n_cpu=8):
    max_cpu = round(cpu_count() * 0.8)
    n_cpu = min(n_cpu, max_cpu)
    n_cpu = max(n_cpu, 1)
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()
    img_d = img_stack.copy()
    pool = Pool(n_cpu)
    res = []
    partial_func = partial(img_denoise_nl_single, patch_size=patch_size, patch_distance=patch_distance)
    for result in tqdm(pool.imap(func=partial_func, iterable=img), total=len(img)):
        res.append(result)
    pool.close()
    pool.join()
    img_d = np.array(res)
    return img_d


def img_fillhole(img, binary_threshold=0.5):
    img_b = img.copy()
    img_b[np.isnan(img_b)] = 0
    img_b[np.isinf(img_b)] = 0
    img_b[img_b > binary_threshold] = 1
    img_b[img_b < 1] = 0

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_fill_holes(img_b, structure=struct).astype(img.dtype)
    img_fillhole = img * mask
    return mask, img_fillhole



def img_dilation(img, binary_threshold=0.5, iterations=2):
    img_b = img.copy()
    img_b[np.isnan(img_b)] = 0
    img_b[np.isinf(img_b)] = 0
    img_b[img_b > binary_threshold] = 1
    img_b[img_b < 1] = 0

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_dilation(img_b, structure=struct, iterations=iterations).astype(img.dtype)
    img_dilated = img * mask
    return mask, img_dilated


def img_erosion(img, binary_threshold=0.5, iterations=2):
    img_b = img.copy()
    img_b[np.isnan(img_b)] = 0
    img_b[np.isinf(img_b)] = 0
    img_b[img_b > binary_threshold] = 1
    img_b[img_b < 1] = 0

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_erosion(img_b, structure=struct, iterations=iterations).astype(img.dtype)
    img_erosion = img * mask
    return mask, img_erosion






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
        print('Increasing padding thickness to: {}'.format(thick))

    img = np.array(img)
    s = np.array(img.shape)

    if thick == 0 or direction > 3 or s.size > 3:
        return img

    hf = np.int32(np.ceil(abs(thick)+1) / 2)  # half size of padding thickness
    if thick > 0:
        if s.size < 3:  # 2D image
            if direction == 0: # padding row
                pad_image = np.zeros([s[0]+thick, s[1]])
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



class IndexTracker(object):
    def __init__(self, ax, X, cmap):
        self.ax = ax
        self._indx_txt = ax.set_title(' ', loc='center')
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[self.ind, :, :], cmap=cmap)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        #self.ax.set_ylabel('slice %s' % self.ind)
        self._indx_txt.set_text(f"frame {self.ind + 1} of {self.slices}")
        self.im.axes.figure.canvas.draw()


def image_movie(data, ax=None, cmap='gray'):
    # show a movie of image in python environment
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    tracker = IndexTracker(ax, data, cmap)
    # monkey patch the tracker onto the figure to keep it alive
    fig._tracker = tracker
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return tracker


def plot3D(data, axis=0, index_init=None, clim=[], cmap='rainbow'):
    fig, ax = plt.subplots()
    if index_init is None:
        index_init = int(data.shape[axis]//2)        
    if len(clim) == 2:
        im = ax.imshow(data.take(index_init,axis=axis), cmap=cmap, clim=clim)
    else:
        im = ax.imshow(data.take(index_init,axis=axis), cmap=cmap)
    fig.subplots_adjust(bottom=0.15)
    axslide = fig.add_axes([0.1, 0.03, 0.8, 0.03])
    im_slider = Slider(
        ax=axslide,
        label='index',
        valmin=0,
        valmax=data.shape[axis] - 1,
        valstep=1,
        valinit=index_init,
    )
    def update(val):
        im.set_data(data.take(val,axis=axis))
        fig.canvas.draw_idle()
   
    im_slider.on_changed(update)
    plt.show()
    return im_slider 


def add_blur_noise_to_img(img, gaussian_kernel=1.5, poisson_counts=[10000]):
    img_a = np.random.poisson(img*poisson_counts[0])/poisson_counts[0]
    img_b = gf(img_a, gaussian_kernel)
    img_c = np.random.poisson(img_b*poisson_counts[-1])/poisson_counts[-1]
    return img_c


def update_img_with_mask_comp(img, mask_comp, method='median'):
    n = len(img)
    n_comp = len(mask_comp)
    img1 = np.zeros_like(img)

    for i in range(n): # ni2, ni3
        t_max = -1
        c = 0
        for j in range(n_comp):
            t = img[i] * mask_comp[j]

            val = t[t > 0]
            if len(val) > 0:
                if method == 'median':
                    t = np.median(val)
                else:
                    t = np.mean(val)
                t_max = max(t, t_max)

                if t < t_max * 1e-3:
                    c += img[i] * mask_comp[j]
                else:
                    c += t * mask_comp[j]
        img1[i] = c
    return img1