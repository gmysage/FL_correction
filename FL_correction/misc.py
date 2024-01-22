import numpy as np


def remove_elem_in_list(elem_type, elem_to_remove):
    elem_type_new = []
    for elem in elem_type:
        if elem_to_remove in elem:
            continue
        elem_type_new.append(elem)
    return elem_type_new


def remove_elem_in_dataset(dataset, element_type, elem_to_remove):
    s = dataset.shape
    n = s[0]
    s_new = (n-1, *s[1:])
    data_new = np.zeros(s_new)
    idx = 0
    for i in range(n):
        if elem_to_remove in element_type[i]:
            continue
        data_new[idx] = dataset[i]
        idx += 1
    return data_new



def remove_elem_in_dict(dict, elem_tom_remove):
    keys = dict.keys()
    dict_new = {}
    idx = 0
    for i, key in enumerate(keys):
        if elem_to_remove in key:
            continue
        dict_new[key] = dict[key]
    return dict_new


def recon_astra_sub(proj, theta, rot_cen=None, method='FBP_CUDA', num_iter=20):
    import tomopy
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