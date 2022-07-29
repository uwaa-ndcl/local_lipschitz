import os
import time
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

import my_config
import bounds_adv
import network_bound 

import mnist as exp
#import cifar10 as exp
#import alexnet as exp
#import vgg16 as exp

"""
# download MNIST data
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(
        root=exp.mnist_dir, transform=transform, train=True, download=True)
loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=1)
for batch_idx, (inputs, targets) in enumerate(loader):
    (batch_size, ch, h, w) = inputs.shape 
    for i in range(batch_size):
        x = inputs[i,:,:,:]
"""


save_npz = os.path.join(exp.main_dir, 'many_local_bounds.npz')
names = ['2', '3', '8']
n = len(names)
bounds = np.full(n, np.nan)
times = np.full(n, np.nan)
eps_list = [.01, .1, 1.0] # what should I use for this, should I vary it?
net = exp.net()
for i in range(n):
    eps = eps_list[i]
    filename = os.path.join(exp.main_dir, names[i]+'.png')
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
