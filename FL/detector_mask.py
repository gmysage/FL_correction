import numpy as np
from numba import njit, prange
from tqdm import trange
from .image_util import rm_nan
from .util import load_dict_to_hdf5, save_dict_to_hdf5

def load_mask3D(fn='mask3D.h5'):
    mask3D = load_dict_to_hdf5(fn)
    return mask3D


def prep_detector_mask3D(alfa=15, beta=60, length_maximum=200, fn_save='mask3D.h5'):
    print('Generating detector 3D mask ...')
    mask = {}
    for i in trange(length_maximum, 6, -1):
        mask[f'{i}'] = generate_detector_mask(alfa, beta, i)
    print(f'Saving {fn_save} ...')
    save_dict_to_hdf5(mask, fn_save)
    return mask


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

    alfa = np.float32(alfa0) / 2 / 180 * np.pi
    theta = np.float32(theta0) / 2 / 180 * np.pi

    N1_0 = np.int16(np.ceil(leng0 * np.tan(alfa)))  # for original matrix
    N2_0 = np.int16(np.ceil(leng0 * np.tan(theta)))

    leng = leng0 + 30
    N1 = np.int16(np.ceil(leng * np.tan(alfa)))
    N2 = np.int16(np.ceil(leng * np.tan(theta)))

    Mask = np.zeros([2 * N1 - 1, 2 * N2 - 1, leng])

    s = Mask.shape
    s0 = (2 * N1_0 - 1, 2 * N2_0 - 1, leng0)  # size of "original" matrix

    M1 = g_mask((s[0], s[2]), alfa, N1)
    M2 = g_mask((s[1], s[2]), theta, N2)
    M1[N1 - 1, :] = 1
    M1[N1 - 1, 0] = 0
    M2[N2 - 1, :] = 1
    M2[N2 - 1, 0] = 0

    Mask1 = Mask.copy()
    Mask2 = Mask.copy()

    for i in range(s[1]):
        Mask1[:, i, :] = M1
    for i in range(s[0]):
        Mask2[i, :, :] = M2

    Mask = Mask1 * Mask2  # element by element multiply
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

    cent = np.array([np.floor(s[0] / 2), np.floor(s[1] / 2)])
    delt = np.array([np.floor(s0[0] / 2), np.floor(s0[1] / 2)])

    xs = np.int16(cent[0] - delt[0])
    xe = np.int16(cent[0] + delt[0] + 1)
    ys = np.int16(cent[1] - delt[1])
    ye = np.int16(cent[1] + delt[1] + 1)
    zs = 0
    ze = leng0

    Mask3D_cut = Mask3D[xs:xe, ys:ye, zs:ze]  # col, slice, row
    Mask3D_cut = np.transpose(Mask3D_cut, [1, 2, 0])  # slice, row, col
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
            i = I + 1
            j = J + 1
            if (np.abs(N - i) >= j * tan_alfa):
                M1[I, J] = 0
                M11[I, J] = M1[I, J]

            elif (np.abs(N - i) < (j - 1) * tan_alfa
                  and (np.abs(N - i) + 1) > j * tan_alfa):

                desi_1 = (j - 1) * tan_alfa - np.floor((j - 1) * tan_alfa)
                desi_2 = j * tan_alfa - np.floor(j * tan_alfa)
                M11[I, J] = 0.5 * (desi_1 + desi_2)
                M1[I, J] = M11[I, J] / cos_alfa

            elif (np.abs(N - i) < j * tan_alfa
                  and (np.abs(N - i) + 1) > j * tan_alfa
                  and np.abs(N - i) > (j - 1) * tan_alfa):

                desi_1 = j * tan_alfa - np.floor(j * tan_alfa)
                M11[I, J] = 0.5 * desi_1 * (desi_1 / tan_alfa)
                M1[I, J] = M11[I, J] / cos_alfa

            elif ((np.abs(N - i) + 1) < j * tan_alfa
                  and np.abs(N - i) < (j - 1) * tan_alfa
                  and (np.abs(N - i) + 1) > (j - 1) * tan_alfa):
                desi_1 = np.ceil((j - 1) * tan_alfa) - (j - 1) * tan_alfa
                M11[I, J] = 1 - 0.5 * desi_1 * (desi_1 / tan_alfa)
                M1[I, J] = M11[I, J] / cos_alfa

            else:
                tmp = np.arctan(1.0 * (N - i) / j)
                M11[I, J] = 1
                M1[I, J] = M11[I, J] / np.cos(tmp)
    return M1


@njit
def g_radial_mask(M_shape, radial_dis, shape_mask):
    M_normal = np.zeros(M_shape)
    dis = np.floor(radial_dis)
    a = np.floor(radial_dis)
    b = radial_dis - a
    for i in prange(1, np.max(dis) + 1):
        # flag_mask = np.zeros(M_shape) # mark the position with radial distance == i
        temp = np.zeros(M_shape)

        s = dis.shape
        for p in prange(s[0]):
            for q in prange(s[1]):
                for r in prange(s[2]):
                    if dis[p, q, r] == i:
                        temp[p, q, r] = temp[p, q, r] + shape_mask[p, q, r] * (1 - b[p, q, r])
                        # flag_mask[p, q, r]=1
                    if dis[p, q, r] == i - 1:
                        temp[p, q, r] = temp[p, q, r] + shape_mask[p, q, r] * b[p, q, r]
                        # flag_mask[p, q, r]=1
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
    if xe > sd[1]: xe = sd[1] + 1

    ys = col - int(np.floor(s[2] / 2))
    ye = ys + s[2]
    if ys < 0: ys = 0
    if ye > sd[2]: ye = sd[2] + 1

    zs = sli - int(np.floor(s[0] / 2))
    ze = zs + s[0]
    if zs < 0: zs = 0
    if ze > sd[0]: ze = sd[0] + 1

    data_mask = data[ze:ze, xs:xe, ys:ye]

    return data_mask


@njit
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
    ys = int(col - int(s[2] / 2))
    ye = int(col + int(s[2] / 2) + 1)
    zs = int(sli - int(s[0] / 2))
    ze = int(sli + int(s[0] / 2) + 1)
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
