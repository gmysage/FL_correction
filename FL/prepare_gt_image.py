from numba import jit, njit, prange
import numpy as np
from tqdm import trange
from skimage import filters
from skimage import morphology
from skimage.morphology import disk

def circle_mask(img, ratio=1, val=0):
    s = img.shape
    if len(s) == 2:
        img = img[np.newaxis]
    s = img.shape
    im = np.zeros_like(img)
    x = np.arange(s[1])
    y = np.arange(s[2])
    X, Y = np.meshgrid(y, x)
    X = X / s[2]
    Y = Y / s[1]
    mask = np.float32(((X-0.5)**2 + (Y-0.5)**2)<(ratio/2)**2)
    mask_minus = 1 - mask
    for i in range(s[0]):
        im[i] = img[i] * mask + (mask_minus) * val
    return np.squeeze(im)


class Ball3D:
    def __init__(self, x, y, z, r, v, g, idx):

        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.v = v
        self.g = g
        self.idx = idx

    def __repr__(self):
        txt0 = f'ball #{self.idx}:'
        txt1 = f'x={self.x:3.2f},  y={self.y:3.2f},  z={self.z:3.2f},  r={self.r:3.2f}'
        txt2 = f'val={self.v},   gradient={self.g}'
        txt = f'\n{txt0}\n{txt1}\n{txt2}\n\n'
        return txt

    def to_dict(self):
        b_dict = {}
        b_dict['x'] = self.x
        b_dict['y'] = self.y
        b_dict['z'] = self.z
        b_dict['r'] = self.r
        b_dict['v'] = self.v  
        b_dict['g'] = self.g      
        b_dict['idx'] = self.idx
        return b_dict

def ball_distance(b1, b2):
    return ((b1.x - b2.x)**2 + (b1.y - b2.y)**2 + (b1.z - b2.z)**2)**0.5



def gen_single_3D_ball(s_3D, val=1, radius_range=[], cen_ratio=0.9, gradient_range=[1,1], idx=0):
    s = np.array(s_3D)
    s_s = np.int16(s * (1 - cen_ratio) / 2)
    s_e = np.int16(s * (1 + cen_ratio) / 2)
    if len(radius_range) == 1:
        r = radius_range[0]
    elif len(radius_range) == 2:
        r = np.random.random() * (radius_range[1] - radius_range[0]) + radius_range[0]
    else:
        r = np.random.random() * 0.3 * np.max(s)
    x = np.random.random() * (s_e[0] - s_s[0]) + s_s[0]
    y = np.random.random() * (s_e[1] - s_s[1]) + s_s[1]
    z = np.random.random() * (s_e[2] - s_s[2]) + s_s[2]

    g = np.sort(np.random.uniform(gradient_range, size=2))
    p = Ball3D(x, y, z, r, val, g, idx=idx)
    return p


def gen_3D_ball(s_3D, n_ball, ball_exist=[], val_range=[],
                radius_range=[], cen_ratio=0.5, gradient_range=[1,1],
                allow_overlap=False):

    ball = ball_exist.copy()
    if len(val_range) != 2:
        vmin, vmax = 0.5, 1
    else:
        vmin, vmax = val_range
    #s = np.array(s_3D.shape)
    s = np.array(s_3D)
    s_s = np.int16(s * (1 - cen_ratio) / 2)
    s_e = np.int16(s * (1 + cen_ratio) / 2)

    if allow_overlap:
        idx = len(ball) + 1
        for i in range(n_ball):
            v = np.random.uniform(vmin, vmax)
            p = gen_single_3D_ball(s_3D, v, radius_range, cen_ratio, gradient_range,idx)
            ball.append(p)
            """
            if len(radius_range) == 1:
                r = radius_range[0]
            elif len(radius_range) == 2:
                r = np.random.random() * (radius_range[1] - radius_range[0]) + radius_range[0]
            else:
                r = np.random.random() * 0.3 * np.max(s)
            x = np.random.random() * (s_e[0] - s_s[0]) + s_s[0]
            y = np.random.random() * (s_e[1] - s_s[1]) + s_s[1]
            z = np.random.random() * (s_e[2] - s_s[2]) + s_s[2]
            p = Ball3D(x, y, z, r, idx=idx)
            ball.append(p)
            """
            idx += 1
    else:
        max_iter = 1000
        i = 0
        while True:
            idx = len(ball) + 1
            v = np.random.uniform(vmin, vmax)
            p = gen_single_3D_ball(s_3D, v, radius_range, cen_ratio, gradient_range, idx)
            overlap = False
            for j in range(len(ball)):
                p0 = ball[j]
                dist = ball_distance(p, p0)
                if dist < p.r + p0.r:
                    overlap = True
                    break
            if not overlap:
                ball.append(p)
            if len(ball) >= n_ball:
                break
            i += 1
            if i >= max_iter:
                break
    return ball, len(ball)



def add_3D_ball(img3D_exist, ball):
    img3D = img3D_exist.copy()
    for i in range(len(ball)):
        b = ball[i]
        img3D = draw_3D_ball_with_gradient(img3D, b.x, b.y, b.z, b.r, b.v, b.g)
        
    return img3D    


def draw_3D_ball_with_gradient(img3D, x, y, z, r, val, g):

    s = img3D.shape
    r0 = int(np.ceil(r))
    xs = int(max(0, x-r0))
    ys = int(max(0, y-r0))
    zs = int(max(0, z-r0))
    xe = int(min(s[0], x + r0))
    ye = int(min(s[1], y + r0))
    ze = int(min(s[2], z + r0))
    X,Y,Z = np.mgrid[xs:xe, ys:ye, zs:ze]
    v0 = val * g[0]
    v1 = val * g[1]
    dis2 = (X-x)**2 + (Y-y)**2 + (Z-z)**2
    ratio2 = dis2 / r**2
    
    g_val = ratio2 * (v1 - v0) + v0 
    g_val[g_val<0] = 0
    g_val[g_val > val] = val
    
    m = dis2 <= r**2
    img3D[xs:xe, ys:ye, zs:ze] += np.float32(m) * g_val
    return img3D


@njit(parallel=True)
def voronoi3D_njit(p, val, gx, gy, gz, x, y, z):
    s = gx * gy * gz
    n = len(p)
    img = np.zeros(gx * gy * gz)
    #ts = time.time()
    for i in prange(s):
        idx = 0
        dis = 100
        for j in prange(n):
            curr_dis = (x[i]-p[j, 0])**2 + (y[i]-p[j, 1])**2 + (z[i]-p[j, 2])**2
            if curr_dis <= dis:
                idx = j
                dis = curr_dis
        img[i] = val[idx]
    #te = time.time()
    return img


def gen_crack_mask(s_3D, n=500, dilation=0, d_range=[0.5, 1]):
    gx, gy, gz = s_3D # e.g., 32, 256, 256
    p = np.random.random([n, 3])
    val = np.random.random(n) 
    val = val * (d_range[1] - d_range[0]) + d_range[0]
    #val = (val +1 )/2 # scale to [0.5, 1]

    x, y, z = np.mgrid[0:gx, 0:gy, 0:gz]
    x = x.flatten()/(gx-1)
    y = y.flatten()/(gy-1)
    z = z.flatten()/(gz-1)
    img3 = voronoi3D_njit(p, val, gx, gy, gz, x, y, z)
    img3 = img3.reshape([gx, gy, gz]) 
    mask = np.zeros_like(img3)
    if dilation > 0:
        footprint = disk(dilation)
    for i in range(gx):
        m = filters.roberts(img3[i])
        m[m > 0] = 1
        if dilation > 0:
            m = morphology.dilation(m, footprint)
        mask[i] = 1 - m
    return mask, img3

def random_select_poly(img3D, n):
    import random
    #from skimage.transform import rescale, resize
    a = list(img3D.flatten())
    val_unique = np.sort(list(set(a)))
    total_num = len(val_unique)
    num = min(total_num-2, n)
    n_unique = len(val_unique)
    id_unique = list(np.arange(n_unique))
    idx = np.int32(random.sample(id_unique, num))


    #idx = np.int32(random.sample(set(val_unique), num))
    mask = np.zeros(img3D.shape)
    mask_temp = mask.copy()
    for i in range(len(idx)):
        mask_temp = np.zeros(img3D.shape)
        #v = idx[i]
        v = val_unique[i]
        mask_temp[np.abs(img3D-v)<1e-6] = 1
        mask = mask + mask_temp
    return mask, img3D*mask, val_unique    

####################################################################
def add_hole(img3D, n_holes_range=[1000, 10000], radius_range=[1, 2]):
    shape3D = img3D.shape
    ball_exist=[]    
    n_ball = np.random.randint(n_holes_range[0], n_holes_range[1])
    m_ball, n = gen_3D_ball(shape3D, 
                              n_ball, 
                              ball_exist,
                              [1, 1], # value_range
                              radius_range, # radius_range
                              1,      #cen_ratio,  
                              [1, 1], # gradient_range,
                              True # allow overlap
                              )
    m = add_3D_ball(img3D_exist, m_ball)
    m[m>0] = 1
    m = 1 - m 
    img3D = img3D * m
    return img3D

def gen_gt_image_sphere():
    fn_root = '/data/FL_correction/FL_2/gt_image2'
    shape3D = (32, 200, 200)
    img3D_exist = np.zeros(shape3D)
    n_ball = 20
    ball_exist=[]
    radius_range=[5, 40]
    cen_ratio=0.9
    img_slice = []
    for i in trange(1000): # 1000 3D structure
        n_ball = np.random.randint(30, 100)
        if np.random.random() > 0.5:
            allow_overlap = True 
        else:
            allow_overlap = False
        g1 = np.random.uniform(-1, 0.5)
        g2 = np.random.uniform(0, 1.5)
        gradient_range = np.sort([g1, g2])
        val_range = np.sort(np.random.uniform(1, 10, size=2))
        ball, n = gen_3D_ball(shape3D, 
                              n_ball, 
                              ball_exist,
                              val_range,
                              radius_range, 
                              cen_ratio,  
                              gradient_range,
                              allow_overlap
                              )
        img3D = add_3D_ball(img3D_exist, ball)

        m = img3D[img3D>0]
        m = np.sort(m)[int(len(m)*0.96)]
        img3D = img3D / m

        # generate hole mask
        t = np.random.random()
        if t > 0.7: 
            img3D = add_hole(img3D, [1000, 10000], [1,1.5])

        t = np.random.random()
        if t > 0.8: # add crack
            num = int(np.random.uniform(500, 10000))
            m_c, im = gen_crack_mask(shape3D, num)
            img3D *= m_c
        
        # apply circle mask
        c_ratio = np.random.uniform(0.6, 0.98)
        img3D = circle_mask(img3D, ratio=c_ratio, val=0)

        img_slice.append(img3D[0])
        fsave = f'{fn_root}/img_gt_{i:04d}.tiff'
        io.imsave(fsave, img3D.astype(np.float32))
    img_slice = np.array(img_slice)
    
def gen_gt_image_poly():
    fn_root = '/data/FL_correction/FL_2/gt_image2'
    idx_start = 1000
    shape3D = (32, 200, 200)
    img_slice2 = []
    for i in trange(1000):
        num = int(np.random.uniform(200, 1000))
        m_c, im = gen_crack_mask(shape3D, num, d_range=[0, 1])
        n = int(np.random.uniform(num/10, num/1.5))
        a, im_s, c = random_select_poly(im, n)
        
        c_ratio = np.random.uniform(0.5, 0.95)
        img3D = circle_mask(im_s, ratio=c_ratio, val=0)
        img3D = img3D / np.max(img3D) /np.random.uniform(0.9, 1.5)
        
        # add holes
        t = np.random.random()
        if t > 0.7: 
            img3D = add_hole(img3D, [1000, 10000], [1,1.5])

        img_slice2.append(img3D[0])
        fsave = f'{fn_root}/img_gt_{i+idx_start:04d}.tiff'
        io.imsave(fsave, img3D.astype(np.float32))
    img_slice2 = np.array(img_slice2) 


def gen_noisy_image():
    fn_gt_root = '/data/FL_correction/FL_2/gt_image'
    fn_gt = np.sort(glob.glob(fn_gt_root + '/*'))
    n = len(fn_gt)
    device = 'cuda'
    for i in trange(n):
        angle_e = int(np.random.uniform(120, 180))
        n_inv = np.random.randint(2, 4)
        angle_list = np.arange(0, angle_e, n_inv)
        fn_id = fn_gt[i].split('/')[-1][:-5].split('_')[-1]
        img3D = io.imread(fn_gt[i])
        prj = re_projection_cuda(img3D, 
                                    angle_list, 
                                    param=None, 
                                    rho_compound=1, 
                                    atten_coef=None,
                                    elem='', 
                                    use_ref=False,
                                    device=device
                                    )
        ph = np.random.randint(1000, 5000)
        C_init = np.ones_like(img3D[np.newaxis])
        prj_n = np.random.poisson(prj* ph)/ph   
        n_iter = np.random.randint(20, 50)                    
        rec = torch_mlem_recon_batch(C_init,  # (n_ref, n_sli, H, W)
                    prj_n,           # (n_angle, n_sli, W)
                    angle_list,     # (n_angle)
                    atten = None,   # (n_angle, n_sli, H, W)
                    em_cs = None,   # (n_angle, n_ref)
                    rho = 1,
                    pix = 1,
                    n_iter = n_iter,
                    beta = 1e-3,
                    delta = 0.01,
                    device = 'cuda'
                    )
        rec = rec[0]
        fsave_rec = f'/data/FL_correction/FL_2/noisy_image/img_noise_{fn_id}.tiff'
        fsave_prj = f'/data/FL_correction/FL_2/proj_image/prj_{fn_id}.tiff'
        fsave_angle = f'/data/FL_correction/FL_2/proj_image/angle_{fn_id}.txt'
        io.imsave(fsave_rec, rec.astype(np.float32))
        io.imsave(fsave_prj, prj_n.astype(np.float32))
        np.savetxt(fsave_angle, angle_list)



def gen_noisy_super_fast_tomo_image():
    fn_gt_root = '/data/FL_correction/FL_2/gt_image2'
    fn_gt = np.sort(glob.glob(fn_gt_root + '/*'))
    n = len(fn_gt)
    device = 'cuda'
    img_slice3 = []
    for i in trange(n):
        angle_e = int(np.random.uniform(150, 180))
        n_inv = 2.5
        d_angle = np.arange(0, n_inv, 0.25) - n_inv/2
        angle_list = np.arange(0, angle_e, n_inv)
        fn_id = fn_gt[i].split('/')[-1][:-5].split('_')[-1]
        img3D = io.imread(fn_gt[i])
        prj_sub = []
        for delta_angle in d_angle:
            prj_d = re_projection_cuda(img3D, 
                                        angle_list+delta_angle, 
                                        param=None, 
                                        rho_compound=1, 
                                        atten_coef=None,
                                        elem='', 
                                        use_ref=False,
                                        device=device
                                        )
            prj_sub.append(prj_d)
        prj_sub = np.array(prj_sub)
        prj = np.sum(prj_sub, axis=0) / len(d_angle)
        ph = np.random.randint(100, 500)
        C_init = np.ones_like(img3D[np.newaxis])
        prj_n = np.random.poisson(prj* ph)/ph   
        n_iter = np.random.randint(20, 30)  

        if i%2 == 0:    # recon using mlem            
            rec = torch_mlem_recon_batch(C_init,  # (n_ref, n_sli, H, W)
                        prj_n,           # (n_angle, n_sli, W)
                        angle_list,     # (n_angle)
                        atten = None,   # (n_angle, n_sli, H, W)
                        em_cs = None,   # (n_angle, n_ref)
                        rho = 1,
                        pix = 1,
                        n_iter = n_iter,
                        beta = 0.1, #1e-3
                        delta = 0.01,
                        device = device
                        )
            rec = rec[0]
        else: # recon using gridrec
            rec = tomopy.recon(prj_n, -angle_list/180.*np.pi, algorithm='gridrec')
            rec[rec<0] = 0
        t = np.random.random()
        if t>0.1:
            ph1 = np.random.randint(50, 100)
            rec_n=np.random.poisson(rec*ph1)/ph1
        else:
            rec_n = rec

        fsave_rec = f'/data/FL_correction/FL_2/noisy_image2_gridrec/img_noise_{fn_id}.tiff'
        fsave_prj = f'/data/FL_correction/FL_2/proj_image2_gridrec/prj_{fn_id}.tiff'
        fsave_angle = f'/data/FL_correction/FL_2/proj_image2_gridrec/angle_{fn_id}.txt'
        io.imsave(fsave_rec, rec_n.astype(np.float32))
        io.imsave(fsave_prj, prj_n.astype(np.float32))
        np.savetxt(fsave_angle, angle_list)

        img_slice3.append(rec_n[16])
    img_slice3 = np.array(img_slice3)