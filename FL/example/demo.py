import matplotlib.pyplot as plt
import pyxas
import time
import numpy as np
from tqdm import trange
from skimage import io
import FL
from skimage.transform import rescale, resize
import glob
import torch

######################################
# 1.1 load ground-truth data
######################################
fn_root = '/data/FL_correction/examples/demo_torch_sparse_xanes'
# read binning 2 data
bining = 1
if bining == 2:
    s3D = (80, 80, 80)
elif bining == 1:
    s3D = (160, 160, 160)


Ni2 = io.imread(fn_root + f'/gt_img/Ni2_gt_bin{bining}.tiff')
Ni3 = io.imread(fn_root + f'/gt_img/Ni3_gt_bin{bining}.tiff')
Ni = Ni2 + Ni3
Mn = io.imread(fn_root + f'/gt_img/Mn_gt_bin{bining}.tiff')
Co = io.imread(fn_root + f'/gt_img/Co_gt_bin{bining}.tiff')

ref1 = np.loadtxt(fn_root + '/ref_NiO_97365.txt')
ref2 = np.loadtxt(fn_root + '/ref_LiNiO2_97366.txt')

angle_list = np.loadtxt(fn_root + '/angle_list_bin2_new.txt')
theta = angle_list / 180. * np.pi
eng_list = np.loadtxt(fn_root + '/energy_list_bin2_new.txt')

######################################
# 1.2 load mask3D
######################################
alfa, beta = 11, 28 
length_maximum = 240
fn_mask = fn_root + f'/mask3D_{int(length_maximum)}.h5'
# mask3D = FL.prep_detector_mask3D(alfa=alfa, beta=beta, length_maximum=length_maximum, fn_save=fn_mask)
mask3D = FL.load_mask3D(fn_mask)

######################################
# 1.3 load param
######################################
param = FL.load_param(fn_root + f'/param_bin{bining}.txt') 
n_elem = param['nelem']
elem_type = param['elem_type'] # Ni, Co, Mn
elem_comp = param['elem_compound'] # LiNiO2, LiCoO2, LiMnO2
pix = param['pix']
eng_list = param['eng_list']

######################################
# 1.4 cross-section of Ni by providing reference spectrum
######################################

# update emission cross-section
param['em_cs'], _, _ = FL.update_emission_xray_cs_with_ref(param['em_cs'], (ref1, ref2), elem='Ni', ref_ratio=[], plot_flag=True)

# update incident x-ray absorption cross-section
param['cs'], _, _ = FL.update_incident_xray_cs_with_ref(param['cs'], (ref1, ref2), elem='Ni', ref_ratio=[], plot_flag=True)

######################################
# 1.5 calculate and save ground-truth attenuation to ./Angle_prj 
######################################

"""
M_NMC = 100 # g/mol
scale = 1e15 # scale unit: molar ---> femto molar
#pix = 2e-5 # from: pix = param['pix'] = 200 nm = 2e-5 cmbechstein piano
pix_mole = rho * pix**3 / M_NMC * scale # pix_mole = 4.63e-2 femto molar
"""
device = 'cuda'
raw3D = {}
raw3D['Ni2'] = Ni2 #* pix_mole
raw3D['Ni3'] = Ni3 #* pix_mole
raw3D['Ni'] = Ni   #* pix_mole
raw3D['Co'] = Co   #* pix_mole
raw3D['Mn'] = Mn   #* pix_mole
raw_gt = FL.pre_treat(list(raw3D[ele] for ele in elem_type))

max_l = Ni.shape[1] # 160
detector_mask = mask3D[f'max_l']
rho = 4.63

# use referece spectrum to interplate the absorption cross-section
dict_use_ref = {}
dict_use_ref['use_ref'] = True
dict_use_ref['ref_3D'] = FL.pre_treat(Ni2, Ni3) # (2, 100, 100, 100)
dict_use_ref['ref_comp'] = 'LiNiO2'

# calculate attenuation and projection on ground-truth
res = FL.cal_frac(raw_gt, scale_range=[0.02, 0.98], enable_scale=True, scale_limit=0.95)
frac_4D = torch.from_numpy(res['frac']).float().to(device)
fsave0 = fn_root + f'/Angle_prj_bin{bining}'
fsave = fn_root + f'/Angle_prj_bin{bining}_test'
# cal_xrf_atten
FL.cal_xrf_atten_at_angles(frac_4D, 
                            param, 
                            angle_list, 
                            detector_mask,
                            detector_offset_angle=0,
                            position_det='r',
                            dict_use_ref=dict_use_ref,
                            fsave=fsave,
                            )
# cal_incident x-ray _atten
FL.cal_xray_atten_at_angles(frac_4D, 
                            param, 
                            angle_list, 
                            dict_use_ref,
                            fsave=fsave,
                            )
# cal and save total atten                            
FL.cal_total_atten_at_angles(fsave, param)


######################################
# 1.6 calculate attenuated projection images at angles 
######################################  

####### add noise to projection ######

fsave = fn_root + f'/Angle_prj_bin{bining}_test'
prj = {}
prj_clean = {}
ph = 4000
for elem in ['Co', 'Mn']:
    att_ele = FL.read_attenuation_at_all_angle(angle_list, fsave, elem, 'all')
    img_ele = raw3D[elem]
    
    prj_clean[elem] = FL.re_projection_cuda(img_ele, 
                                    angle_list, 
                                    param, 
                                    rho, 
                                    att_ele, 
                                    elem=elem, 
                                    use_ref=False)
    prj[elem] = np.random.poisson(prj_clean[elem]* ph)/ph

for elem in ['Ni']:
    att_ele = FL.read_attenuation_at_all_angle(angle_list, fsave, elem, 'all')
    img_ele = FL.pre_treat(Ni2, Ni3)
    prj_clean[elem] = FL.re_projection_cuda(img_ele, 
                                    angle_list, 
                                    param, 
                                    rho, 
                                    att_ele, 
                                    elem=elem, 
                                    use_ref=True)
    prj[elem] = np.random.poisson(prj_clean[elem]* ph)/ph

# save ground-truth XRF projection image
for ele in elem_type:
    #fsave_prj = fn_root + f'/gt_img/xrf_prj_gt_{ele}_bin{bining}.tiff'
    fsave_prj = fn_root + f'/gt_img/xrf_prj_gt_{ele}_bin{bining}_noise.tiff'
    io.imsave(fsave_prj, prj[ele].astype(np.float32))

############################################################################

############################
## init recon (using mlem_cuda_batch)
############################
#device = 'cuda:2'
recon = {0:{}}
t_iter = []
rho = 4.63
pix = param['pix']
n_iter = 200
beta = 0.5 #default: 1e-3, noise_free: 0.2, noise:0.5 
delta = 0.005 # default: 0.01, noise_free: 0.01

# load prj
prj = {}
for ele in elem_type:
    #fn = fn_root + f'/gt_img/xrf_prj_gt_{ele}_bin{bining}.tiff'
    fn = fn_root + f'/gt_img/xrf_prj_gt_{ele}_bin{bining}_noise.tiff'
    prj[ele] = io.imread(fn)

for elem in ['Co', 'Mn']:
    C_init = np.ones((1, *s3D))
    prjs = prj[elem]
    atten = None
    #em_cs = param['em_cs'][elem]
    em_cs = FL.atten_util.extract_em_cs_elem(param, elem, use_ref=False) # in "atten_util.py"

    

    rec = FL.torch_mlem_recon_batch(C_init,     # (n_ref, n_sli, H, W)
                                prjs,        # (n_angle, n_sli, W)
                                angle_list,  # (n_angle)
                                atten,       # (n_angle, n_sli, H, W)
                                em_cs,       # (n_angle, n_ref)
                                rho,
                                pix,
                                n_iter,
                                beta,
                                delta,
                                device
                                )
    recon[0][elem] = rec.squeeze()

ts = time.time()
for elem in ['Ni']: # Ni2, Ni3
    C_init = np.ones((2, *s3D))
    prjs = prj[elem]
    atten = None
    em_cs = FL.atten_util.extract_em_cs_elem(param, 'Ni', use_ref=True)
    #device = 'cuda'

    rec = FL.torch_mlem_recon_batch(C_init,     # (n_ref, n_sli, H, W)
                                prjs,        # (n_angle, n_sli, W)
                                angle_list,  # (n_angle)
                                atten,       # (n_angle, n_sli, H, W)
                                em_cs,       # (n_angle, n_ref)
                                rho,
                                pix,
                                500,
                                beta,
                                delta,
                                device
                                )
    recon[0]['Ni2'] = rec[0]
    recon[0]['Ni3'] = rec[1]
    recon[0]['Ni'] = rec[0] + rec[1]
te = time.time()
t_iter.append(te-ts)
save_recon(recon, fn_root, idx=None) 


###### iterations ####

n_iter_Co_Mn = 200
n_iter_Ni = 1000
beta = 0.5 #default: 1e-3, noise_free: 0.2, noise:0.5 
delta = 0.005 # default: 0.01, noise_free: 0.01

model_3DUNet = FL.load_default_3DUNet_model()
#device = 'cuda'

for it in range(1, 3):
    print(f'{"#"*40}\n{" "*12}iter = {it} \n{"#"*40}')
    recon[it] = {}
    it_pre = it - 1

    dict_use_ref = {}
    dict_use_ref['use_ref'] = True
    dict_use_ref['ref_3D'] = FL.pre_treat(recon[it_pre]['Ni2'], recon[it_pre]['Ni3']) # (2, 100, 100, 100)
    dict_use_ref['ref_comp'] = 'LiNiO2'


    rec_iter = FL.pre_treat(list(recon[it_pre][ele] for ele in elem_type))
    res = FL.cal_frac(rec_iter, scale_range=[0.02, 0.98], enable_scale=True, scale_limit=0.95)
    frac_4D_it = torch.tensor(res['frac'], dtype=torch.float, device=device)
    
    fsave = fn_root + f'/Angle_prj_bin{bining}_{it}'

    # cal_xrf_atten
    FL.cal_xrf_atten_at_angles(frac_4D_it, 
                                param, 
                                angle_list, 
                                detector_mask,
                                detector_offset_angle=0,
                                position_det='r',
                                dict_use_ref=dict_use_ref,
                                fsave=fsave,
                                )
    # cal_incident x-ray _atten
    FL.cal_xray_atten_at_angles(frac_4D_it, 
                                param, 
                                angle_list, 
                                dict_use_ref,
                                fsave=fsave,
                                )
    # cal and save total atten                            
    FL.cal_total_atten_at_angles(fsave, param)

    for elem in ['Co', 'Mn']:
        print(f'{"#"*40}\n{" "*6}recon {elem} at iteration #{it} \n{"#"*40}')
        #C_init = np.ones((1, 80, 80, 80))
        C_init = recon[it_pre][elem][np.newaxis]
        prjs = prj[elem]
        atten = FL.read_attenuation_at_all_angle(angle_list, fsave, elem, 'all')

        #em_cs = param['em_cs'][elem]
        em_cs = FL.atten_util.extract_em_cs_elem(param, elem, use_ref=False)
        rec = FL.torch_mlem_recon_batch(C_init,     # (n_ref, n_sli, H, W)
                                    prjs,        # (n_angle, n_sli, W)
                                    angle_list,  # (n_angle)
                                    atten,       # (n_angle, n_sli, H, W)
                                    em_cs,       # (n_angle, n_ref)
                                    rho,
                                    pix,
                                    n_iter_Co_Mn,
                                    beta,
                                    delta,
                                    device
                                    )
        for i in range(len(rec)):
            rec[i] = FL.denoise_3d(rec[i], model_3DUNet, device=device)
        recon[it][elem] = rec[0]

    ts = time.time()
    for elem in ['Ni']: # Ni2, Ni3
        print(f'{"#"*40}\n{" "*6}recon {elem} at iteration #{it} \n{"#"*40}')
        C_init = FL.pre_treat(recon[it_pre]['Ni2'], recon[it_pre]['Ni3'])
        prjs = prj[elem]
        atten = FL.read_attenuation_at_all_angle(angle_list, fsave, elem, 'all')
        em_cs = FL.atten_util.extract_em_cs_elem(param, 'Ni', use_ref=True)
        #device = 'cuda'

        rec = FL.torch_mlem_recon_batch(C_init,     # (n_ref, n_sli, H, W)
                                    prjs,        # (n_angle, n_sli, W)
                                    angle_list,  # (n_angle)
                                    atten,       # (n_angle, n_sli, H, W)
                                    em_cs,       # (n_angle, n_ref)
                                    rho,
                                    pix,
                                    n_iter_Ni,         # n_iter = 1000
                                    beta,
                                    delta,
                                    device
                                    )
        
        for i in range(len(rec)):
            rec[i] = FL.denoise_3d(rec[i], model_3DUNet, device=device)

        recon[it]['Ni2'] = rec[0]
        recon[it]['Ni3'] = rec[1]
        recon[it]['Ni'] = rec[0] + rec[1]
    te = time.time()
    t_iter.append(te-ts)
    save_recon(recon, fn_root, None)

# each recon on Ni (160, 160, 160) (1000 EM steps) takes about 250 sec (~6min)
# each recon on Mn/Co (160, 160, 160) (100 EM steps) takes about 25 sec (~0.5 min)
# each XRF atten calucation takes about 190 sec (3min, 10sec)
# each X_ray atten calculation takes about 30 sec (0.5 min)
# so overall, each iteration takes ~ 11 min



######################### with 3D UNet for Ni2, Ni3, ###########################
#device = 'cuda'

fn_recon = fn_root +'/recon'
recon, n_iter = FL.load_recon(fn_recon) 

it_pre = 1
cur_recon = recon[it_pre]
cur_recon['Ni'] = cur_recon['Ni2'] + cur_recon['Ni3']  

rec_iter = FL.pre_treat(list(cur_recon[ele] for ele in elem_type))
res = FL.cal_frac(rec_iter, scale_range=[0.02, 0.98], enable_scale=True, scale_limit=0.95)
frac_4D_it = torch.tensor(res['frac'], dtype=torch.float, device=device)
dict_use_ref = {}
dict_use_ref['use_ref'] = True
dict_use_ref['ref_3D'] = FL.pre_treat(recon[it_pre]['Ni2'], recon[it_pre]['Ni3']) # (2, 100, 100, 100)
dict_use_ref['ref_comp'] = 'LiNiO2'


C_init = FL.pre_treat(cur_recon['Ni2'], cur_recon['Ni3'])
prjs = prj['Ni']
em_cs = FL.atten_util.extract_em_cs_elem(param, 'Ni', use_ref=True)
rho = 4.63
pix = param['pix']
C_cuda = torch.from_numpy(C_init).float().to(device)
I_cuda = torch.from_numpy(prjs).float().to(device)
em_cs_cuda = torch.from_numpy(em_cs * rho * pix).float().to(device)
theta_cuda = torch.from_numpy(theta).float().to(device)

f_xrf = fn_root + f'/Angle_prj_bin{bining}_{it_pre}'
atten_xrf = FL.read_attenuation_at_all_angle(angle_list, f_xrf, 'Ni', 'fl')
atten_xrf_cuda = torch.from_numpy(atten_xrf).float().to(device) 

atten_xray = FL.read_attenuation_at_all_angle(angle_list, f_xrf, 'Ni', 'xray')  # save folder as xrf
atten_xray_cuda = torch.from_numpy(atten_xray).float().to(device) 
atten_cuda = atten_xrf_cuda * atten_xray_cuda

beta = 0.5
delta=1e-3


"""
### freeze denoiser
for p in model_3DUNet.parameters():
    p.requires_grad = False

### unfreeze last layer
unfreeze_last_layer(model_3DUNet)
"""

model_3DUNet = load_default_3DUNet_model(device=device)
model_3DUNet.freeze_encoder()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_3DUNet.parameters()),
    lr=1e-5
) 

C_hist = []
loss_hist = {'mse':[], 'poisson':[], 'tv':[], 'total':[]}
C_cuda = torch.from_numpy(C_init).float().to(device)

C = C_cuda.clone()
flag_update_atten = True

n_update_atten = 20 

for it in trange(100):
    print(f'Iter : {it}  ')
    model_3DUNet.train()
    optimizer.zero_grad(set_to_none=True)
    # cal_incident x-ray _atten
    if (it+1) % n_update_atten == 0:
        f_xray = fn_root + '/Angle_prj_tmp'
        FL.cal_xray_atten_at_angles(frac_4D_it, 
                                    param, 
                                    angle_list, 
                                    dict_use_ref,
                                    f_xray
                                    )
        atten_xray = FL.read_attenuation_at_all_angle(angle_list, f_xray, 'Ni', 'xray')  
        atten_xray_cuda = torch.from_numpy(atten_xray).float().to(device) 
        atten_cuda = atten_xrf_cuda * atten_xray_cuda

        #flag_update_atten = False
    
        C = FL.mlem_cuda_batch(C_cuda,     # (n_ref, n_sli, H, W)
                                atten_cuda,  # (n_angle, n_sli, H, W)
                                em_cs_cuda,  # (n_angle, n_ref)
                                theta_cuda,  # (n_angle)
                                I_cuda,      # (n_angle, n_sli, W)
                                20,
                                beta,
                                delta
                                )
    
    #t = C.detach()
    #t = t[t>0.01]
    #scale = torch.sort(t)[0][int(len(t)*0.99)]
    scale = torch.tensor(1.0).float().to(device)
    C.requires_grad_()
    C_unet = C.unsqueeze(1) # (n_ref, 1, n_sli, H, W)
    #scale = torch.max(C_unet)
    C_unet = C_unet / scale
    ref_idx = torch.zeros(C_unet.shape[0], dtype=torch.long, device=device)
    C_d = model_3DUNet(C_unet, ref_idx).squeeze() * scale

    Pf = FL.forward_emission_batch_autograd(atten_cuda, C_d, em_cs_cuda, theta_cuda)
    Pf.clamp_min_(0)

    loss_mse = torch.nn.functional.mse_loss(Pf, I_cuda)
    loss_poisson = FL.poisson_loss_image(Pf, I_cuda)
    loss_tv = FL.gradient_loss(C_d)

    loss = 1e3 * loss_mse + 1 * loss_poisson + 0.1 * loss_tv
    print(f'Iter: {it} | '
          f'total loss: {loss.item():.5e} | '
          f'mse loss: {loss_mse.item():.5e} | '
          f'tv loss: {loss_tv.item():.5e} | '
          f'poisson loss: {loss_poisson.item():.4e}\n\n'
          )
    loss_hist['mse'].append(loss_mse.item())
    loss_hist['poisson'].append(loss_poisson.item())
    loss_hist['tv'].append(loss_tv.item())
    loss_hist['tv'].append(loss_tv.item())
    loss_hist['total'].append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    C_cpu = C_d.detach().cpu().numpy()
    if (it+1) % n_update_atten == 0:

        C_cpu = C_d.detach().cpu().numpy()
        cur_recon['Ni'] = np.sum(C_cpu, axis=0)
        rec_iter = FL.pre_treat(list(cur_recon[ele] for ele in elem_type))
        res = FL.cal_frac(rec_iter, scale_range=[0.02, 0.99], enable_scale=True, scale_limit=0.99)
        frac_4D_it = torch.from_numpy(res['frac']).float().to(device)
        dict_use_ref['ref_3D'] = C_cpu

    C_hist.append(C_cpu[0, 80])
    if (it) % n_update_atten == 0:
    #if True:
        p = Pf.detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.suptitle(f'iter = {it:02d}')
        plt.subplot(221);plt.imshow(C_cpu[0, 80], clim=[0, 0.62])
        plt.title('rec');plt.colorbar()
        plt.subplot(222);plt.imshow(C_cpu[0, 80]-Ni2[80], clim=[-0.08, 0.08], cmap='Spectral_r')
        plt.title('rec - gt');plt.colorbar()
        plt.subplot(223);plt.imshow(p[30]-prjs[30], clim=[-0.012, 0.012], cmap='bwr')
        plt.title('Pf - prj_noise');plt.colorbar()
        plt.subplot(224);plt.imshow(p[30]-prj_clean['Ni'][30], clim=[-0.012, 0.012], cmap='bwr')
        plt.title('Pf - prj_clean');plt.colorbar()
        plt.pause(1)

plt.figure()
plt.subplot(221);plt.plot(np.array(loss_hist['total']), label='total');plt.legend()
plt.subplot(222);plt.plot(np.array(loss_hist['tv'])*0.1, label='tv');plt.legend()
plt.subplot(223);plt.plot(np.array(loss_hist['poisson'])*1, label='poisson');plt.legend()
plt.subplot(224);plt.plot(np.array(loss_hist['mse'])*1e3, label='mse');plt.legend()


############################################################################
### test and debug
ll = {'Ni':0.5, 'Mn':0, 'Co':0, 'Ni2':0, 'Ni3':0}
hl = {'Ni':0.75, 'Mn':0.2, 'Co':0.35, 'Ni2':0.65, 'Ni3':0.55}
for it in range(3):
    plt.figure(figsize=(15, 12))
    plt.suptitle(f'iter = {it}')

    id1 = 1
    for elem in elem_type:
        plt.subplot(3, 3, id1); plt.imshow(raw3D[elem][80], clim=[ll[elem], hl[elem]])
        plt.colorbar();plt.title(f'{elem}: ground-truth')
        id1 += 1

        plt.subplot(3, 3, id1); plt.imshow(recon[it][elem][80], clim=[ll[elem], hl[elem]])
        plt.colorbar();plt.title(f'{elem}: recon iter {it}')
        id1 += 1

        plt.subplot(3, 3, id1); plt.imshow(recon[it][elem][80]-raw3D[elem][80], cmap='coolwarm', clim=[-0.05, 0.05])
        plt.colorbar();plt.title(f'{elem} Error')
        id1 += 1

for it in range(3):
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'iter = {it}')
    id1 = 1
    for elem in ['Ni2', 'Ni3']:
        plt.subplot(2, 3, id1); plt.imshow(raw3D[elem][80], clim=[ll[elem], hl[elem]])
        plt.colorbar();plt.title(f'{elem}: ground-truth')
        id1 += 1

        plt.subplot(2, 3, id1); plt.imshow(recon[it][elem][80], clim=[ll[elem], hl[elem]])
        plt.colorbar();plt.title(f'{elem}: recon iter {it}')
        id1 += 1

        plt.subplot(2, 3, id1); plt.imshow(recon[it][elem][80]-raw3D[elem][80], cmap='coolwarm', clim=[-0.05, 0.05])
        plt.colorbar();plt.title(f'{elem} Error')
        id1 += 1


