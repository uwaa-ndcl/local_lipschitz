import os
import time
import glob
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm

import my_config
import bounds_adv
import network_bound 

import mnist as exp
#import cifar10 as exp
#import alexnet as exp
#import vgg16 as exp

device = my_config.device
net = exp.net()
net = net.to(device)
num_workers = 8

# imagenet
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.all_imgs = glob.glob('/home/trevor/Downloads/inet_1100_224/*.png')

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        img = Image.open(img_path)
        img = exp.transform(img)
        #img = img.to(device)
        target = -1 # this doesn't matter
        return img, target

# dataset and loader
if exp.net_name == 'mnist':
    dataset = torchvision.datasets.MNIST(
        root=exp.main_dir, train=False, download=True, transform=exp.transform)

elif exp.net_name == 'cifar10':
    dataset = torchvision.datasets.CIFAR10(
        root=exp.main_dir, train=False, download=True, transform=exp.transform)
elif (exp.net_name == 'alexnet') or (exp.net_name == 'vgg16'):
    #dataset = torchvision.datasets.ImageNet(
        #root=exp.main_dir, train=False, download=True, transform=exp.transform)
    dataset = CustomImageDataset()

loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        #dataset)

# create arrays
#n = len(loader)
#n = 101 # number of images to test
n = 11 # number of images to test
eps = np.random.uniform(.1,5,n)
bounds = np.full(n, np.nan)
times = np.full(n, np.nan)

# find bound for each input
with torch.no_grad():
    for i, (img, target) in enumerate(tqdm(loader,total=n)):
    #for i, (img, target) in enumerate(loader)):
    #for (img, target) in loader:
        if i==n:
            break
        img, target = img.to(device), target.to(device)
        t0 = time.time()
        bounds[i] = network_bound.local_bound(net, img, eps[i], batch_size=32)
        t1 = time.time()
        times[i] = t1 - t0

# save results
save_npz = os.path.join(exp.main_dir, 'many_bounds.npz') 
np.savez(save_npz, eps=eps, bounds=bounds, times=times)

# print results
times = times[1:] # remove 1st trial as it may have additional computational overhead
time_min = np.min(times)
time_max = np.max(times)
time_avg = np.average(times)
print('min time', time_min)
print('max time', time_max)
print('avg time', time_avg)
