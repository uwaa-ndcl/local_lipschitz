import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import network_bound
import my_config

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.maxpool = nn.MaxPool2d(2) 
        self.fc1 = nn.Linear(16*4*4,84)
        self.fc2 = nn.Linear(84,10)
        self.relu = nn.ReLU(inplace=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# these functions are only needed to run C&W attacks
normalize = transforms.Normalize(0.0, 1.0)
unnormalize = lambda x: x*1.0 + 0.0

# create network
net = MyNet()
net.to(my_config.device)
relu = torch.nn.ReLU(inplace=False)
flatten = nn.Flatten()
net.layers = [net.conv1, relu,
              net.maxpool,
              net.conv2, relu,
              net.maxpool,
              flatten,
              net.fc1, relu,
              net.fc2]

# nominal input
x0 = torch.rand(1,1,28,28)
print(x0)
x0 = x0.to(my_config.device)

# input perturbation size and batch size
eps = 10**-3
batch_size = 10**3

# calculate global Lipschitz bound
layer_bounds = network_bound.global_bound(net, x0)
glob_bound = np.prod(layer_bounds)
print('GLOBAL LIPSCHITZ UPPER BOUND')
print('bound:', glob_bound)

# calculate local Lipschitz bound
bound = network_bound.local_bound(net, x0, eps, batch_size=batch_size)
print('\nLOCAL LIPSCHITZ UPPER BOUND')
print('epsilon:', eps)
print('bound:', bound)

# calculate adversarial bound using local Lipschitz constant
print('\nADVERSARIAL LOWER BOUND')
n_runs = 10
eps_min = 1e-4
eps_max = 1e-1
y0 = net(x0)
top2, ind_top2 = torch.topk(y0.flatten(), 2)
delta = (top2[0]-top2[1]).item()
for i in range(n_runs):
    eps_i = (eps_max + eps_min)/2
    bound = network_bound.local_bound(net, x0, eps_i, batch_size=batch_size)
    if eps_i*bound < delta/np.sqrt(2):
        eps_greatest = eps_i
        #print('eps', eps_i, 'lower bound')
        #print('epsilon:', eps_i)
        eps_min = eps_i
    else:
        #print('eps', eps_i, 'NOT lower bound')
        #print('bound', bound)
        eps_max = eps_i

print('largest epsilon for lower bound is', eps_greatest)
