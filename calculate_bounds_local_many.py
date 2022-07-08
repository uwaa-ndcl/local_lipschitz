import os
import time
import numpy as np
from PIL import Image
import torch

import my_config
import bounds_adv
import network_bound 

import mnist as exp
#import cifar10 as exp
#import alexnet as exp
#import vgg16 as exp

save_npz = os.path.join(exp.main_dir, 'many_local_bounds.npz')
names = ['2', '3', '8']
n = len(names)
bounds = np.full(n, np.nan)
times = np.full(n, np.nan)
eps = .01 # what should I use for this, should I vary it?
net = exp.net()
for i, name in enumerate(names):
    filename = os.path.join(exp.main_dir, name+'.png')
    x0 = Image.open(filename)
    x0 = exp.transform_test(x0)
    x0 = torch.unsqueeze(x0, 0)
    x0 = x0.to(my_config.device)
    t0 = time.time()
    bounds[i] = network_bound.local_bound(net, x0, eps, batch_size=32)
    t1 = time.time()
    times[i] = t1 - t0

print(times)
np.savez(save_npz, names=names, bounds=bounds, times=times)
