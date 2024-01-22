import pyxas
import xraylib
import glob
from scipy.signal import medfilt as mf
from skimage import io
from tqdm import trange
from skimage.transform import rescale

'''
discard Si
'''
'''
1. normalize by ion-chamber
2. normalize by XRF_cross_section and mass_density ---> unit of cm
3. multiply by pixel area ---> unit of cm^3
4. multiply by mass_density ---> unit of gram
5. normalize by molar_mass ---> unit of molar
6. scale ---> unit of femto-molar
'''
global mask3D

fn_root = '/data/absorption_correction/example/XRF_from_APS/sample_2023Q3_HXN/data_no_Si'
fn = fn_root + '/everything.h5'
f = h5py.File(fn, 'r') # 'data', 'elements', 'names', 'thetas'

# read all elements: ['Si', 'Ti', 'Cr', 'Fe', 'Ni', 'Ba', 'TFY', 'ic']
img_all = np.array(f['data']) # (8, 53, 301, 601)
elements_type = ['Si', 'Ti', 'Cr', 'Fe', 'Ni', 'Ba', 'TFY', 'ic']

# remove 'Si'
elem_to_remove = 'Si'
img_all = FL.remove_elem_in_dataset(img_all, elements_type, elem_to_remove)

# image binning for attenuation calculation
b = 4 # binning factor
s = img_all.shape
img_all = img_all[:, :, :s[2]//b*b, :s[3]//b*b] # (8, 53, 300, 600)

# read angle list
angle_list = np.array(f['thetas']) 
theta = angle_list / 180.0 * np.pi

# need to take " negative angle " for tomopy reconstruction
theta_tomopy = - theta

n_theta = len(theta)
f.close()

# load parameters
param = FL.load_param(fn_root + '/param.txt')
n_elem = param['nelem']
elem_type = param['elem_type']
pix = param['pix']
em_cs = param['em_cs']
M = param['M']
rho = param['rho']
cs = FL.get_atten_coef(param['elem_type'], param['XEng'], param['em_E'])

# start correction on "binning reconstruction"
param['pix'] *= b  # change pixel size in "param"
scale = 1e15 # scale unit: molar ---> femto molar


# read projection images
proj = {}
proj['ic'] = img_all[-1]
for i, ele in enumerate(elem_type):
    proj[ele] = img_all[i] / proj['ic']             # step1: ormalize by ion-chamber
    proj[ele] = proj[ele] / em_cs[ele] / rho[ele]   # step2: normalize by XRF_cross_section and mass_density
    proj[ele] = proj[ele] * pix**2                  # step3: multiply by pixel area
    proj[ele] = proj[ele] * rho[ele]                # step4: multiply by mass_density
    proj[ele] = proj[ele] / M[ele]                  # step5: normalize by molar_mass
    proj[ele] = proj[ele] * scale                   # step6: scale to femto-molar
    proj[ele] = proj[ele][:, :, 100:500]


# binning projection image
proj_raw = FL.pre_treat(list(proj[ele] for ele in elem_type))
s_proj_raw = proj_raw.shape
proj_bin = FL.bin_ndarray(proj_raw, (s_proj_raw[0], s_proj_raw[1], 
                                     s_proj_raw[2]//b, s_proj_raw[3]//b), 'sum')

########################################################
# reconstruct the raw data and generate detector mask3D
########################################################

rec3D = {}
# reconstruction at full size using MLEM
for i, ele in enumerate(elem_type):
    print(f'recontructing {ele}:')
    rec3D[ele] = FL.recon_astra_sub(proj[ele], theta_tomopy, method='EM_CUDA', num_iter=16)
recon_raw = FL.pre_treat(list(rec3D[ele] for ele in elem_type))
FL.save_recon(fn_root, recon_raw, elem_type, -2) # "-2" means saving the file with filenames as "Ni_iter_-2.tiff"

# bining the reconstruction and save it
s_rec_raw = recon_raw.shape
recon_bin = FL.bin_ndarray(recon_raw, (s_rec_raw[0], s_rec_raw[1]//b, 
                                     s_rec_raw[2]//b, s_rec_raw[3]//b), 'sum')
FL.save_recon(fn_root, recon_bin, elem_type, -1)  # "-1" means saving the file with filenames as "Ni_iter_-1.tiff"                                   

##########################################
# generate 3D mask (if not generated yet)
##########################################

length_maximum = 200
fn_mask = fn_root + f'/mask3D_{int(length_maximum)}.h5'
mask3D = FL.prep_detector_mask3D(alfa=20.6, theta=20.6, length_maximum=length_maximum, fn_save=fn_mask) 

####################################
# (optional) read raw reconstruction
####################################
recon_raw = FL.read_recon_all_elem(fn_root, -2, elem_type)   # full size
recon_bin = FL.read_recon_all_elem(fn_root, -1, elem_type)   # binned

##############################
# load 3D mask
##############################
length_maximum = 200
fn_mask = fn_root + f'/mask3D_{int(length_maximum)}.h5'
mask3D = FL.load_mask3D(fn_mask)


##############################
# start iterative correction
##############################
num_cpu = 25
recon_cor = FL.read_recon_all_elem(fn_root, -1, elem_type) # read current reconstruction 

ref_prj = proj_bin # read binned projection (sinogram)

FL.load_global_mask(mask3D) # set mask3D as global parameter

# start iteration
for it in range(1, 5):
    ts = time.time()    
    # calculate attenuation
    print(f'calculate attenuation on iteration {it}')  
    recon_cor = FL.smooth_filter(recon_cor, 3)  # smooth the image
    recon_cor = FL.rm_boarder(recon_cor, 5)     # optional: remove redundent (noisy) features on the image edge by 5 pixels at each side

    fsave_iter =  fn_root + f'/Angle_prj_{it:02d}'
    FL.cal_and_save_atten_prj(param, cs, recon_cor, angle_list, ref_prj, 
                        fsave=fsave_iter, align_flag=False, 
                        enable_scale=False, num_cpu=num_cpu)
    
    te = time.time() 
    print(f'calculate attenucation takes {te-ts:4.1f} seconds\n')  

    # start correction
    for i, elem in enumerate(elem_type):
        print(f'correction on {elem}')
        ref_tomo = np.ones(recon_cor[i].shape)                          

        save_tiff = True
        fpath_save = fn_root + '/recon'
        n_iter = 16
        
        '''
        # option 1: run in CPU
        num_cpu_recon = 4 # do not change
        cor = FL.absorption_correction_mpi(elem, ref_tomo, angle_list, fsave_iter, 
                                        n_iter, num_cpu_recon, save_tiff, fpath_save)
        '''
        # option 2: run in GPU(numba cuda)
        cor = FL.cuda_absorption_correction_wrap(elem, ref_tomo, angle_list, fsave_iter, 
                                            n_iter, save_tiff, fpath_save)
        recon_cor[i] = FL.rm_boarder(cor, 5)
    FL.save_recon(fn_root, recon_cor, elem_type, it)
    te2 = time.time()
    print(f'reconstruction takes {te2-te:4.1f} seconds\n')


##########################################################
####### rescale attenuation to full size (witout binign)
##########################################################

n_angle = len(angle_list)
b = 4
iter_id = 4
elem_type = ['Ti', 'Cr', 'Fe', 'Ni', 'Ba']

# generate rescaled attenuation coefficient
# files will be saved in "fsave_rescale"
for elem_id, elem in enumerate(elem_type):
    ts = time.time()
    fsave_rescale = fn_root + '/Angle_prj_-10'
    fpath_atten = fn_root + f'/Angle_prj_{iter_id:02d}'
    for i in trange(n_angle):
        coef_att = FL.read_attenuation(i, fpath_atten, elem)
        coef_scale = rescale(coef_att, b)
        FL.write_attenuation(elem, coef_scale, angle_list[i], i, fsave_rescale)
        FL.write_projection('m', elem, proj_raw[elem_id, i], angle_list[i], i, fsave_rescale)


###############################
## correction on full_size image
###############################

#elem_type = ['Ti', 'Cr', 'Fe', 'Ni', 'Ba']

fpath_recon = fn_root + '/recon'  # path which stored the raw reconstruction 
fsave_atten = fn_root + '/Angle_prj_-10'  # path which saves the rescaled attenuation files

res3D_all = {}
n_iter=16
fpath_atten = fn_root + '/Angle_prj_-10'
save_tiff = True 

for elem in elem_type:
    print(f'process {elem}')
    ref_tomo = FL.read_recon(fpath_recon, -2, elem)
    '''
    # CPU version
    num_cpu = 4
    cor = FL.absorption_correction_mpi(elem, ref_tomo, angle_list, fpath_atten,
                                        n_iter, num_cpu, save_tiff, fpath_save)
    '''
    # GPU version
    cor = FL.cuda_absorption_correction_wrap(elem, ref_tomo, angle_list, fpath_atten, 
                                            n_iter, save_tiff, fpath_save)
    res3D_all[elem] = cor.copy()
    
recon_full = np.zeros(recon_raw.shape)
for i, elem in enumerate(elem_type):
    recon_full[i] = res3D_all[elem]

# save the files in folder fn_root/recon'
# filename is, e.g., "Fe_iter_-10.tiff"
FL.save_recon(fn_root, recon_full, elem_type, iter_id=-10) 



#######################################



def recon_astra_sub(proj, theta, rot_cen=None, method='FBP_CUDA', num_iter=20):
    if rot_cen is None:
        rot_cen = proj.shape[-1] / 2
    if method=='EM_CUDA':
        extra_options = {}
    else:
        extra_options = {'MinConstraint': 0, }
    options = {'proj_type': 'cuda', 
                'method': method, 
                'num_iter': num_iter,
                'extra_options': extra_options
                }  

    recon = tomopy.recon(proj,
                     theta,
                     center=rot_cen,
                     algorithm=tomopy.astra,
                     options=options,
                     ncore=4)
    recon[recon<0] = 0
    return recon

