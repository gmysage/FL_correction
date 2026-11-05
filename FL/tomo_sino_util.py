import numpy as np
from skimage.transform import warp_polar, rotate
from skimage.util import img_as_float
from scipy.ndimage import geometric_transform
from numpy import sin, cos
from skimage.transform import warp
from scipy import ndimage
import os
from skimage import io
from skimage.transform import iradon, iradon_sart, radon
import glob
from tqdm import trange, tqdm
import porespy as ps
from multiprocessing import Pool, cpu_count
from functools import partial
#from image_util import load_image_stack, save_image_stack
from skimage.transform import AffineTransform, warp
from scipy.ndimage import gaussian_filter1d
import random
import FL
from FL import poly2D_image
from FL.image_util import image_value_substitution, non_uniform_deform_image
#######################################

def demo_script_sino_tomo():
    fn_root = '/media/mingyuan/Data/dataset'
    prefix = 'DV2K/DV2K_512'
    prefix_stack =['DV2K/DV2K_512',
            'ImagePairs/bin2_512',
            'random_shapes/random_512',
            #'porespy/porespy_512',
            'brain_tumor/tumor_512',
            #'cracks/ball_crack',
            #'cracks/fine_crack',
             ]
    for prefix in prefix_stack:
        print(f'processing {prefix}')
        fn_stack = fn_root + '/' + prefix + '_' + 'gt'
        #img_stack = load_image_stack(fn_stack, 'tiff', output_size=(512, 512))
        img_stack = FL.load_image_stack(fn_stack, 'tiff', output_size=(200, 200))
        # img_stack_m = img_stack
        # img_stack_m = circ_mask(img_stack, 0, 0.9)
        # img_stack_m = circ_mask_random(img_stack, 0, [0.5, 0.98])

        img_stack_m, m_stack = FL.circ_mask_random_deform(img_stack, [0.2, 0.9], 9)
        img_poly = FL.generate_poly_bkg_stack(img_stack.shape, order=3, z_range=(0.4, 1.5))
        img_stack_m = img_stack_m * img_poly
        #theta = np.arange(0, 180, 0.5)
        d_theta = np.arange(-0.75, 0.75, 0.25)
        theta = np.arange(-70, 70, 1) # missing angle
        d_theta = [0]
        counts = 10000
        n_cpu = 16
        sino_stack = sino_blur_img_stack(img_stack_m, theta, d_theta, None, counts, n_cpu)
        rec_stack = iradon_stack(sino_stack, theta, n_cpu)
        #rec_stack = FL.img_denoise_bm3d_mpi(rec_stack, sigma=0.05, non_neg=True)
        rec_stack[rec_stack<0] = 0
        #fsave = fn_root + '/' + prefix + '_' + 'blur/blur_img'
        fsave_dir = fn_root + '/' + prefix + '_' + 'blur_missing_angle'
        os.makedirs(fsave_dir, exist_ok=True)
        fsave = fsave_dir + '/blur_img' # for missing angle
        FL.save_image_stack(rec_stack, prefix=fsave, start_id=0)

        # save binned raw image
        fsave_dir = fn_root + '/' + prefix + '_' + 'raw_image'
        fsave = fsave_dir + '/raw_img'  # for missing angle
        os.makedirs(fsave_dir, exist_ok=True)
        FL.save_image_stack(img_stack_m, prefix=fsave, start_id=0)

def demo_join_folder():
    fn_root = '/media/mingyuan/Data/dataset'
    prefix_stack =['DV2K/DV2K_512',
            'ImagePairs/bin2_512',
            'random_shapes/random_512',
            'porespy/porespy_512',
            'brain_tumor/tumor_512',
            'cracks/ball_crack',
            'cracks/fine_crack',
             ]
    prefix_stack = ['DV2K/DV2K_512',
                    'random_shapes/random_512',
                    'ImagePairs/bin2_512',
                    'brain_tumor/tumor_512',
                    ]
    folders_gt = []
    folders_bl = []
    for prefix in prefix_stack:
        print(f'processing {prefix}')
        f_gt = fn_root + '/' + prefix + '_' + 'raw_image'
        f_bl = fn_root + '/' + prefix + '_' + 'blur_missing_angle'
        folders_bl.append(f_bl)
        folders_gt.append(f_gt)
    # cd /media/mingyuan/Data/dataset/recon_missing_angle/gt_image
    FL.join_img_from_folder(folders_gt, n_file_in_folder=[], file_format='tiff', save_prefix='gt_img')
    # cd /media/mingyuan/Data/dataset/recon_missing_angle/blur_image
    FL.join_img_from_folder(folders_bl, n_file_in_folder=[], file_format='tiff', save_prefix='blur_img')

def demo_train_missing_angle():

    gt_dir = '/media/mingyuan/Data/dataset/recon_missing_angle/gt_image'
    noisy_dir = '/media/mingyuan/Data/dataset/recon_missing_angle/blur_image'
    FL.train_rrdb(gt_dir, noisy_dir, epochs=100)


def demo_generate_recon_with_varied_rotation_center():
    fn_root = '/media/mingyuan/Data/dataset'
    prefix_stack =[#'DV2K/DV2K_512',
                   #'ImagePairs/bin2_512',
                    'random_shapes/random_512',
                    #'balls/balls',
                    'brain_tumor/tumor_512',
                     ]
    global_id = 0
    fsave_root = fn_root + '/recon_at_rot_cen'
    for prefix in prefix_stack:
        print(prefix)
        fn_stack = fn_root + '/' + prefix + '_' + 'gt'
        img_stack = FL.load_image_stack(fn_stack, 'tiff', output_size=(512, 512))
        n = len(img_stack)
        angles = np.arange(0, 180)
        for i in trange(0, n, 10):
            c_shift = np.random.randint(-40, 40, 10)
            c_shift = np.sort(c_shift)
            for j in range(10):
                cen = c_shift[j]
                shifts = (256 + cen, 256)
                sino = generate_sinogram(img_stack[i], shifts, angles).T
                rec = iradon(sino, angles)
                fsave = fsave_root + f'/img_{global_id:06d}_{cen}.tiff'
                io.imsave(fsave, rec.astype(np.float32))
                global_id = global_id + 1

        sino_stack = sino_blur_img_stack(img_stack, angles, d_theta, None, counts, n_cpu)
        rec_stack = iradon_stack(sino_stack, theta, n_cpu)
        rec_stack[rec_stack<0] = 0


#####################################
def rotate_image_skimage(image, center, theta_degrees):
    """Rotates an image around a specific center point using scikit-image."""
    theta_radians = np.deg2rad(theta_degrees)
    cx, cy = center

    # Define transformations
    shift_to_origin = AffineTransform(translation=(-cx, -cy))
    rotation = AffineTransform(rotation=-theta_radians)  # Inverse for inverse warping
    shift_back = AffineTransform(translation=(cx, cy))

    # Combine transformations
    combined_transform = shift_to_origin + rotation + shift_back

    # Warp image (ensuring values remain floating point for accurate summation later)
    rotated_image = warp(image, combined_transform, mode='constant', cval=0, preserve_range=True)
    return rotated_image


def generate_sinogram(image, center, angles):
    """
    Rotates an image at multiple angles, sums the pixel columns,
    and assembles the results into a 2D matrix.

    Parameters:
        image (numpy.ndarray): The 2D input image.
        center (tuple): The (x, y) coordinates of the rotation center.
        angles (iterable): A series of angles in degrees (e.g., range(0, 181)).

    Returns:
        numpy.ndarray: A 2D array where each row corresponds to an angle.
    """
    # 1. Convert angles list/range into a standard NumPy array
    angles_arr = np.asarray(angles)
    num_angles = len(angles_arr)

    # 2. Determine the width of the 1D projection (axis=0 sum yields columns, length = image width)
    # Note: axis=0 sums down the vertical columns, producing a 1D array equal to the image width.
    num_columns = image.shape[1]

    # 3. Pre-allocate the 2D output matrix (Shape: Number of Angles x Image Width)
    output_matrix = np.zeros((num_angles, num_columns), dtype=np.float32)

    # 4. Loop through each angle, rotate, and save the row sum
    for idx, angle in enumerate(angles_arr):
        img_rot = rotate_image_skimage(image, center, angle)

        # Calculate the projection (summing across axis=0)
        output_matrix[idx, :] = np.sum(img_rot, axis=0)

    return output_matrix


####################################
def avg_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def sino_smooth(sino, box_pts=3):
    sino_m = np.zeros(sino.shape) #(pix, angle)
    s = sino.shape
    for i in range(s[1]):
        sino_m[:, i] = avg_smooth(sino[:, i], box_pts)
    return sino_m


def mk_directory(fpath):
    if not os.path.isdir(fpath):
        os.makedirs(fpath)

def gen_noise_sino(sino, counts, mask=None):
    if mask is None:
        mask = 1
    sino_m = sino/np.max(sino) * mask * counts
    sino_n = np.random.poisson(sino_m) / mask / counts * np.max(sino)
    return sino_n


def sino_blur_mpi(fn, theta, d_theta, sino_shape, mask, counts, fpath, fpath_sino_blur, prefix='sino_blur', n_cpu=8):
    num_cpu = n_cpu
    pool = Pool(num_cpu)
    partial_func = partial(sino_blur_mpi_sub, d_theta=d_theta,
                    theta=theta, sino_shape=sino_shape, counts=counts,
                    mask=mask, fpath_sino_blur=fpath_sino_blur, prefix=prefix)
    res = []
    for result in tqdm(pool.imap(func=partial_func, iterable=fn), total=len(fn)):
        res.append(result)
    pool.close()
    pool.join()
    '''
    pool.map(partial(sino_blur_mpi_sub, d_theta=d_theta,
                    theta=theta, sino_shape=sino_shape, counts=counts,
                    mask=mask, fpath_sino_blur=fpath_sino_blur), fn)
    '''


def sino_blur_mpi_sub(fn, d_theta, theta, sino_shape, counts, mask, fpath_sino_blur, prefix):
    s = sino_shape
    img = io.imread(fn)
    n_blur = len(d_theta)
    sino_blur = gen_blur_sino(img, counts, theta, d_theta, mask)
    idx = fn.split('/')[-1].split('.')[0].split('_')[-1]
    #print(idx)
    fsave = f'{fpath_sino_blur}/{prefix}_{idx}.tiff'
    io.imsave(fsave, sino_blur.astype(np.float32))
    return idx

def sino_blur_img_stack(img_stack, theta, d_theta, mask, counts, n_cpu):
    num_cpu = n_cpu
    pool = Pool(num_cpu)
    partial_func = partial(gen_blur_sino, counts=counts, theta=theta, d_theta=d_theta, mask=mask)
    res = []
    for result in tqdm(pool.imap(func=partial_func, iterable=img_stack), total=len(img_stack)):
        res.append(result)
    pool.close()
    pool.join()
    res = np.array(res)
    return res


def gen_blur_sino(img, counts, theta, d_theta, mask):
    """
    theta: in unit of degrees
    """
    n_theta = len(theta)
    n_blur = len(d_theta)
    s = img.shape # img: is a tomography image, sino.shape=(s[1], n)
    sino_blur_sub = np.zeros([n_blur, s[0], n_theta])
    for i in range(n_blur):
        sino_blur_sub[i] = radon(img, theta+d_theta[i])
        sino_blur_sub[i] = gen_noise_sino(sino_blur_sub[i], counts, mask)
    sino_blur = np.sum(sino_blur_sub, axis=0) / n_blur
    return sino_blur

def gaussian_1d(sigma, mu, num, norm_flag=0):
    x = np.linspace(-1, 1, num)
    a = 1./np.sqrt(2 * np.pi)
    b = (x - mu)**2 / sigma**2
    y = a * np.exp(-.5 * b)
    if norm_flag:
        y = y / np.max(y)
    return y


def generate_deformed_circle(shape=(200, 200), radius=100, max_deformation=30, smoothness=8):
    """
    Generates a binary image of a circle deformed strictly inwards.

    Parameters:
    - shape (tuple): The size of the output image (height, width).
    - radius (int): The original radius of the circle.
    - max_deformation (int): Maximum pixels the boundary can deform inwards.
    - smoothness (float): Standard deviation for Gaussian kernel. Higher = smoother.
    """
    height, width = shape
    center_y, center_x = height // 2, width // 2

    # 1. Create a grid of coordinates centered at (0,0)
    y, x = np.ogrid[-center_y:height - center_y, -center_x:width - center_x]

    # 2. Convert Cartesian grid coordinates to polar coordinates (r, theta)
    pixel_r = np.sqrt(x ** 2 + y ** 2)
    pixel_theta = np.arctan2(y, x)  # Returns values between -pi and pi

    # 3. Generate smooth, periodic random noise for the boundary deformation
    # We sample 360 points corresponding to angles from -pi to pi
    num_angles = 360
    angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)

    # Generate random negative values to force inward deformation
    raw_noise = -np.random.uniform(0, max_deformation, size=num_angles)

    # Wrap the noise periodically before smoothing to prevent a visible seam at -pi/pi
    padded_noise = np.tile(raw_noise, 3)
    smoothed_padded = gaussian_filter1d(padded_noise, sigma=smoothness)
    smoothed_noise = smoothed_padded[num_angles:2 * num_angles]

    # 4. Map the smoothed noise back to the image grid based on pixel angles
    # Map pixel_theta from [-pi, pi] to an index in our noise array [0, 359]
    angle_indices = ((pixel_theta + np.pi) / (2 * np.pi) * num_angles).astype(int)
    angle_indices = np.clip(angle_indices, 0, num_angles - 1)

    # 5. Calculate the new dynamic boundary radius for every pixel coordinate
    dynamic_radius = radius + smoothed_noise[angle_indices]

    # 6. Create the binary mask (1 inside the deformed shape, 0 outside)
    mask = (pixel_r < dynamic_radius).astype(np.uint8)

    return mask

def circ_mask_random_deform(img, ratio_range=[0.4, 1], smoothness=8):
    s = img.shape
    if len(s) == 2:
        im_m = img[np.newaxis]
    else:
        im_m = img.copy()
    s = im_m.shape
    m_stack = np.zeros_like(im_m)
    for i in trange(im_m.shape[0]):
        r = np.random.uniform(ratio_range[0], ratio_range[1])
        r = min(r, 1)
        radius = min(s[1], s[2]) // 2
        max_deformation = int(radius * (1-r))
        mask = generate_deformed_circle(shape=(s[1], s[2]),
                                        radius=radius,
                                        max_deformation=max_deformation,
                                        smoothness=smoothness
                                        )
        im_m[i] = im_m[i] * mask
        m_stack[i] = mask
    return im_m, m_stack


def circ_mask_random(img, axis, ratio_range=[0.8, 1], val=0):
    s = img.shape
    if len(s) == 2:
        im_m = img[np.newaxis]
    else:
        im_m = img.copy()
    for i in trange(im_m.shape[0]):
        #ratio = np.random.uniform(ratio_range[0], ratio_range[1])
        r1 = np.random.uniform(ratio_range[0], ratio_range[1])
        r2 = np.random.uniform(ratio_range[0], ratio_range[1])
        im_m[i] = circ_mask(im_m[i], axis=0, ratio=[r1, r2], val=0)
    return im_m


def circ_mask(img, axis, ratio=[1, 1], val=0):
    im = np.float32(img)
    s = im.shape
    if len(s) == 2:
        #m = _get_mask(s[0], s[1], ratio)
        m = _get_mask_asym(s[0], s[1], ratio[0], ratio[1])
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int32) * im + m_out
    else:
        im = im.swapaxes(0, axis)
        dx, dy, dz = im.shape
        #m = _get_mask(dy, dz, ratio)
        m = _get_mask_asym(dy, dz, ratio[0], ratio[1])
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int32) * im + m_out
        im_m = im_m.swapaxes(0, axis)
    return im_m

def _get_mask(dx, dy, ratio):
    """
    Calculate 2D boolean circular mask.

    Parameters
    ----------
    dx, dy : int
        Dimensions of the 2D mask.

    ratio : int
        Ratio of the circle's diameter in pixels to
        the smallest mask dimension.

    Returns
    -------
    ndarray
        2D boolean array.
    """
    rad1 = dx / 2.
    rad2 = dy / 2.
    if dx > dy:
        r2 = rad1 * rad1
    else:
        r2 = rad2 * rad2
    y, x = np.ogrid[0.5 - rad1:0.5 + rad1, 0.5 - rad2:0.5 + rad2]
    return x * x + y * y < ratio * ratio * r2


def _get_mask_asym(dx, dy, rx, ry):
    """
    Calculate 2D boolean circular mask.

    Parameters
    ----------
    dx, dy : int
        Dimensions of the 2D mask.

    ratio : int
        Ratio of the circle's diameter in pixels to
        the smallest mask dimension.

    Returns
    -------
    ndarray
        2D boolean array.
    """
    rad1 = dx / 2.
    rad2 = dy / 2.
    if dx > dy:
        r2 = rad1 * rad1
    else:
        r2 = rad2 * rad2
    y, x = np.ogrid[0.5 - rad1:0.5 + rad1, 0.5 - rad2:0.5 + rad2]
    return (x * x)/(rx * rx) + (y * y)/(ry * ry) < r2


def linear_polar_mapping_2D(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesion to polar coordinates

    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle

    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords

def to_polar_2D(img, center=None, radius=None, output_shape=None,**kwargs):
    if center is None:
        center = (np.array(img.shape)[:2] / 2) - 0.5
    if radius is None:
        w, h = np.array(img.shape)[:2] / 2
        radius = np.sqrt(w ** 2 + h ** 2)
    if output_shape is None:
        height = 360
        width = int(np.ceil(radius))
        output_shape = (height, width)
    else:
        #output_shape = np.int(output_shape)
        height = output_shape[0]
        width = output_shape[1]
    k_radius = width / radius
    k_angle = height / (2 * np.pi)
    #k_angle = np.linspace(0, 2*np.pi, height)
    warp_args = {'k_angle': k_angle, 'k_radius': k_radius, 'center': center}
    warped = warp(img, linear_polar_mapping_2D, map_args=warp_args,
                  output_shape=output_shape)
    return warped


def linear_cart_mapping_2D(coords, center, angle_scale=1):
    '''
    coords: [col, row]
    '''
    x = coords[:, 0] - center[0]
    y = coords[:, 1] - center[1]
    angle = np.arctan2(y, x) / np.pi * 180 * angle_scale -1
    angle[angle<0] = angle[angle<0] + 360 * angle_scale - 1

    radius = np.sqrt(x**2 + (y)**2)
    coords = np.column_stack((radius, angle))
    return coords


def to_linear_2D(img_p, center_offset=None, output_shape=None, angle_scale=1, **kwargs):
    s = img_p.shape
    if output_shape is None:
        w = floor(s[1]/np.sqrt(2))
        output_shape = (2*w, 2*w)
    else:
        w = min(output_shape)/2
    if center_offset is None:
        center = (w-0.5, w-0.5)
    else:
        center = (w-0.5+center_offset[0], w-0.5+center_offset[1])
    warp_args = {'center':center, 'angle_scale':angle_scale}
    warped = warp(img_p, linear_cart_mapping_2D, map_args=warp_args,output_shape=output_shape)
    return warped

def rot3D(img, rot_angle):
    img = np.array(img)
    img = rm_abnormal(img)
    s = img.shape
    if len(s) == 2:    # 2D image
        img_rot = ndimage.rotate(img, rot_angle, order=1, reshape=False)
    elif len(s) == 3:  # 3D image, rotating along axes=0
        img_rot = ndimage.rotate(img, rot_angle, axes=[1,2], order=1, reshape=False)
    elif len(s) == 4:  # a set of 3D image
        img_rot = np.zeros(img.shape)
        for i in range(s[0]):
            img_rot[i] = ndimage.rotate(img[i], rot_angle, axes=[1,2], order=1, reshape=False)
    else:
        raise ValueError('Error! Input image has dimension > 4')
    img_rot[img_rot < 0] = 0
    return img_rot


def noisy(noise_typ,image, mean=0, val=0.1):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = mean
        var = val
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
      # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

       # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy


def rm_abnormal(d):
    t = d.copy()
    t[np.isnan(t)] = 0
    t[np.isinf(t)] = 0
    return t

def iradon_stack(sino_stack, theta, n_cpu):
    num_cpu = n_cpu
    pool = Pool(num_cpu)
    partial_func = partial(iradon, theta=theta)
    rec = []
    for result in tqdm(pool.imap(func=partial_func, iterable=sino_stack), total=len(sino_stack)):
        rec.append(result)
    pool.close()
    pool.join()
    rec = np.array(rec)
    return rec

def demo_generate_shepp_logan():
    import skimage as ski
    from skimage.transform import rescale, resize
    fsave_root = '/media/mingyuan/Data/dataset/shepp_deform/shepp_deform_gt'
    orig_phantom = ski.data.shepp_logan_phantom() # shape=(400, 400)
    img_phantom = rescale(orig_phantom,0.5)
    img_phantom = orig_phantom[::2, ::2]
    mask = np.float32(img_phantom>0)
    mask[50:150, 50:150] = 1
    img_phantom[img_phantom==0] = 0.6
    img_phantom *= mask
    img_phantom = np.round(img_phantom, 2)
    img_phantom[img_phantom==0.1] = 0.45
    val = set(img_phantom.flatten())
    val = np.sort(list(val)) # [0, 0.2, 0.3, 0.45, 0.6, 1]
    n_val = len(val)
    n = 20000
    img_stack = np.zeros((n, 200, 200))
    for i in trange(n):
        val_old = val[i%n_val]
        val_new = np.round(np.random.uniform(-1, 1), 2)
        if val_new <= 0:
            val_new = 0.02
        img = image_value_substitution(img_phantom, val_old, val_new, 0.05) * mask
        # deform the image
        max_deformation = int(np.random.uniform(6, 15))
        smoothness = int(np.random.uniform(15, 20))
        img_d = non_uniform_deform_image(img, max_deformation, smoothness)

        # rotate the image
        ang = int(np.random.uniform(-90, 90))
        img_r = FL.rot3D(img_d, ang)

        # generate slop surface and normalize to the range of (0.6, 1.2)
        order = np.random.choice([2,3,4,5])
        z = poly2D_image((200,200), order=3)
        z1, z2 = np.min(z), np.max(z)
        z = (z-z1) / (z2-z1) * (1.2-0.6) + 0.6

        img_r *= z
        img_r[img_r<0] = 0
        img_stack[i] = img_r
        fsave = fsave_root + f'/gt_image_{i:05d}.tiff'
        io.imsave(fsave, img_r.astype(np.float32))


def generate_missing_angle_tomo():
    f_root = '/media/mingyuan/Data/dataset/shepp_deform'
    fread_root = f_root + '/shepp_deform_gt'
    fsave_root = f_root + '/shepp_deform_blur_mlem'
    fn_list = np.sort(glob.glob(fread_root + '/*.tiff'))
    n = len(fn_list)
    rec_stack = []
    img_stack = []
    for i in trange(n):
        img = io.imread(fn_list[i])
        img_stack.append(img)
        if np.random.rand() < 0.9:
            n_miss_ang = int(np.random.uniform(60, 80))
        else:
            n_miss_ang = 0
        range_ang = np.arange(0, 180-n_miss_ang)
        offset_ang = np.random.randint(-180, 0)
        angle_list = range_ang + offset_ang
        counts = np.random.randint(200, 1000)
        """
        # use iradon method
        sino = gen_blur_sino(img, counts, angle_list, d_theta=[0], mask=None)
        rec = iradon(sino, theta=angle_list)
        """
        # use MLEM method
        sino = FL.re_projection_cuda(img[np.newaxis], angle_list, None)
        sino = np.random.poisson(sino * counts) / counts
        rec = FL.torch_mlem_recon(np.ones((1, 200, 200)), sino[:, 0], angle_list, n_iter=100)[0]

        rec[rec<0] = 0
        fsave = fsave_root + f'/blur_image_{i:05d}.tiff'
        io.imsave(fsave, rec.astype(np.float32))
        rec_stack.append(rec)
    rec_stack = np.array(rec_stack)

def demo_script_old():
    shape = [512, 512]

    ratio = 0.95
    n_img = 1000
    counts = 1500
    porosity = np.random.rand(n_img)
    porosity = 0.7 * porosity + 0.3 # scales to [0.6, 1]
    blob = np.random.rand(n_img)
    blob = (3) * blob + 0 #scales to [1, 5]



    for i in trange(n_img):
        im = ps.generators.blobs(
            shape=shape,
            porosity=porosity[i],
            blobiness=blob[i]
            )
        ls_t = ps.filters.local_thickness(im)
        ls_t = circ_mask(ls_t, 0, 0.95)
        ls_t = ls_t / np.max(ls_t)
        fsave = f'{fpath_tomo_gt}/tomo_gt_{i:04d}.tiff'
        io.imsave(fsave, ls_t.astype(np.float32))


    # generate sino_gt
    fn = glob.glob(f'{fpath_tomo_gt}/*')
    fn = sort(fn)
    n_theta = 72
    theta = np.linspace(0, 180, n_theta, endpoint=False)

    sino_max = np.zeros(n_img)
    for i in trange(len(fn)):
        img = io.imread(fn[i])
        sino = radon(img, theta)
        sino_max[i] = np.max(sino)
        fsave = f'{fpath_sino_gt}/sino_gt_{i:04d}.tiff'
        io.imsave(fsave, sino.astype(np.float32))

    # generate sino_blur
    s = sino.shape
    sino_shape = s
    y_gau = gaussian_1d(0.36, 0, s[0], 1)
    mask = np.expand_dims(y_gau, 1)
    mask = np.repeat(mask, s[1], 1)


    #d_theta = [0, 0.5, 2, 1.5, 2]
    #d_theta = np.arange(12) * 0.25
    d_theta = np.arange(10) * 0.25
    d_theta0 = [0] * 5
    fn = glob.glob(f'{fpath_tomo_gt}/*')
    fn = np.sort(fn)
    sino_blur_mpi(fn, theta, d_theta, sino_shape, mask, counts, fpath, fpath_sino_blur, 'sino_blur', 16)
    sino_blur_mpi(fn, theta, d_theta0, sino_shape, mask, counts, fpath, fpath_sino_noise, 'sino_noise')

    # recon from sino_blur or sino_noise
    fn = glob.glob(f'{fpath_sino_blur}/*')
    #fn = glob.glob(f'{fpath_sino_noise}/*')
    fn = sort(fn)
    for i in trange(len(fn)):
        sino = io.imread(fn[i])
        #sino = sino_smooth(sino, 3)
        rec_t = iradon(sino, theta)
        rec_t = circ_mask(rec_t, 0, 0.95)
        dice = np.random.random()
        if dice > 0.5:
            rec_t = noisy('gauss', rec_t, 0, 0.03)
        rec_t = rot3D(rec_t, d_theta[-1]/2) # for fbp, rotate positive, # for tomopy mlem, rotate negative
        rec_t[rec_t<0] = 0
        fsave = f'{fpath_tomo_blur}/tomo_blur_{i:04d}.tiff'
        #fsave = f'{fpath_tomo_noise}/tomo_noise_{i:04d}.tiff'
        io.imsave(fsave, rec_t.astype(np.float32))


    # generate polar_tomo_blur
    fn = glob.glob(f'{fpath_tomo_blur}/*')
    fn = sort(fn)
    for i in trange(len(fn)):
        img = io.imread(fn[i])
        s = img.shape
        img_polar = to_polar_2D(img, radius=s[0]/2, output_shape=[720, s[0]/2])
        fsave = f'{fpath_tomo_polar_blur}/tomo_polor_blur_{i:04d}.tiff'
        io.imsave(fsave, img_polar.astype(np.float32))


    # generate polar_tomo_gt
    fn = glob.glob(f'{fpath_tomo_gt}/*')
    fn = sort(fn)
    for i in trange(len(fn)):
        img = io.imread(fn[i])
        s = img.shape
        img_polar = to_polar_2D(img, radius=s[0]/2, output_shape=[720, s[0]/2])
        fsave = f'{fpath_tomo_polar_gt}/tomo_polor_gt_{i:04d}.tiff'
        io.imsave(fsave, img_polar.astype(np.float32))



    ######################################################

    idx = 152
    ff = '/home/gemy/Work/machine_learn/super_sino/256x256'
    rec_blur = io.imread(f'{ff}/test/rec_blur_{idx}.tiff')
    rec_noise = io.imread(f'{ff}/test/rec_noise_{idx}.tiff')
    rec_gt = io.imread(f'{ff}/test/rec_gt_{idx}.tiff')

    theta_long = np.linspace(0, 180, 360, endpoint=False)
    n_theta_long = len(theta_long)
    sino_blur = radon(rec_blur, theta)

    sino_blur_raw = io.imread(f'{ff}/sino_blur/sino_blur_{idx:04d}.tiff')
    sino_gt = io.imread(f'{ff}/sino_gt/sino_gt_{idx:04d}.tiff')
    sino_noise = io.imread(f'{ff}/sino_noise/sino_noise_{idx:04d}.tiff')

    s = rec_blur.shape
    m = np.zeros((s[1], s[1], s[1]), dtype=np.float32)

    for i in range(s[1]):
        m[i, :, i] = 1
    mask = np.zeros((div_num, s[1], s[1], s[1]), dtype=np.float32)
    for i in range(div_num):
        print(i)
        mask[i] = rot3D(m, d_theta[i])
        mask[i] = circ_mask(mask[i], axis=0, ratio=1)

    div_num = len(d_theta)
    m_sum = np.sum(mask, axis=0)
    m_sum = m_sum / div_num
    m_diff = m_sum - m

    sino_correction = np.zeros_like(sino_blur_raw) # shape=(img_width, n_theta)
    for i in trange(n_theta):
        img_r = rot3D(rec_blur, -theta[i])
        img_diff = m_diff * img_r
        img_diff_sino = np.sum(np.sum(img_diff, axis=-1), axis=-1)
        sino_correction[:, i] = sino_blur_raw[:, i] - img_diff_sino


    rec_de_blur = iradon(sino_correction, theta)
    io.imsave(f'rec_de_blur_{idx}.tiff', rec_de_blur.astype(np.float32))
    #################################

    idx = 600
    img = io.imread(f'/home/gemy/Work/machine_learn/super_sino/256x256/tomo_blur/tomo_blur_{idx:04d}.tiff')
    img_gt = io.imread(f'/home/gemy/Work/machine_learn/super_sino/256x256/tomo_gt/tomo_gt_{idx:04d}.tiff')
    sino_blur = io.imread(f'/home/gemy/Work/machine_learn/super_sino/256x256/sino_blur/sino_blur_{idx:04d}.tiff')

    img_tomo = io.imread(f'/home/gemy/Work/machine_learn/super_sino/256x256/test_from_model/o_{idx}_tomo.tiff')
    img_polar = io.imread(f'/home/gemy/Work/machine_learn/super_sino/256x256/test_from_model/o_{idx}_polar.tiff')

    s = sino_blur.shape
    sino_blur_sub = np.zeros([n_blur-1, s[0], s[1]])
    for j in range(n_blur-1):
        sino_blur_sub[j] = radon(img, theta+d_theta[j+1])
    sino_de_blur = sino_blur * n_blur - np.sum(sino_blur_sub, axis=0)



