import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks
from PIL import Image

import my_config
import utils

#import tiny as exp 
import mnist as exp

# original network
# WHY DOESNT IT WORK DOING UNNORMALIZE WITH TRANSFORMS?
net = exp.net()
x0_nml = exp.x0
x0_01_lambda = exp.unnormalize(x0_nml)
x0_01_transforms = exp.unnormalize_transforms(x0_nml)
x0_01_transforms2 = exp.unnormalize_transforms2(x0_nml)
print('try', torch.norm(x0_01_transforms - x0_01_lambda))
x0_01 = x0_01_lambda

# network with inputs on interval [0,1]
net_01 = exp.net()
net_01 = nn.Sequential(exp.normalize, net)
#net_01.layers.insert(0, exp.transform_normalize)
y0_nml = net(x0_nml)
y0_01 = net_01(x0_01)
print('nml v 01 for y err:', torch.norm(y0_01 - y0_nml))

# utils function
print('\nUTILS FUNCTION')
x_attack_success_nml, attack_classes, diffs, min_diff = utils.cw_attack(exp, c=1e0, kappa=0, steps=1000, lr=0.01)
print('attack classes:', attack_classes)
print('n successful attacks:', len(diffs))
print('min attack diff:', min_diff) 

# true index
print('\nMANUAL METHOD')
class_true = torch.topk(y0_nml.flatten(), 1)[1].item()
n = 10
labels = torch.linspace(0, n-1, n, dtype=torch.int64)
#n = 1
#labels = torch.randint(0,10,(1,))
#labels[0] = 8
#print('labels:', labels)

# run the attack
attack = torchattacks.CW(net_01, c=1e0, kappa=0, steps=3000, lr=0.01)
attack_images_01 = attack(x0_01, labels)
Y_01 = net_01(attack_images_01)
attack_classes = torch.topk(Y_01, 1)[1].flatten().tolist()
print('attack classes:', attack_classes)
success_bool = np.not_equal(attack_classes, class_true)
#success_bool[3] = True
success_inds = np.where(success_bool)[0]
n_found = len(success_inds)
print('n successful attacks:', n_found)

# iterate over all successful attacks
diffs = np.full(n_found, np.nan)
for i,success_ind in enumerate(success_inds):
    print('attack ind:', success_ind)
    print('attack class:', attack_classes[success_ind])
    xa_01 = attack_images_01[success_ind,:,:,:]
    xa_nml = exp.normalize(xa_01)
    print('true class:', class_true)

    print('\nUNNORMALIZED [0,1] VALUES')
    print('x0_01 min:', torch.min(x0_01))
    print('x0_01 max:', torch.max(x0_01))
    print('xa_01 min:', torch.min(xa_01))
    print('xa_01 max:', torch.max(xa_01))
    print('diff 01:', torch.norm(xa_01 - x0_01))

    print('\nNORMALIZED VALUES')
    print('x0_nml min:', torch.min(x0_nml))
    print('x0_nml max:', torch.max(x0_nml))
    print('xa_nml min:', torch.min(xa_nml))
    print('xa_nml max:', torch.max(xa_nml))
    diff = torch.norm(xa_nml - x0_nml)
    print('diff nml:', diff)
    diffs[i] = diff

print('\nALL SUCCESSFUL ATTACKS')
print('diffs:', diffs)
