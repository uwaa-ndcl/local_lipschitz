import os
from PIL import Image
import torch

import my_config
import bounds_adv

#import mnist as exp
#import cifar10 as exp
#import alexnet as exp
import vgg16 as exp

# go once
save_npz = os.path.join(exp.main_dir, 'adv_bounds.npz') 
#bounds_adv.compute_bounds(exp, save_npz)
bounds_adv.compute_bounds(exp, save_npz, compute_local=False, compute_global=False, compute_fgsm=True, compute_ifgsm=False, compute_gradasc=False, compute_cw=False)
'''
# go multi
names = ['2', '3', '8']
for name in names: 
    filename = os.path.join(exp.main_dir, name+'.png')
    x0 = Image.open(filename)
    x0 = exp.transform(x0)
    x0 = torch.unsqueeze(x0, 0)
    x0 = x0.to(my_config.device)
    save_npz = os.path.join(exp.main_dir, name+'.npz') 
    exp.x0 = x0
    bounds_adv.compute_bounds(exp, save_npz)
'''
