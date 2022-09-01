import os
import torch

import my_config
from networks import alexnet
import utils


###############################################################################
# lipEstimation spectral norms
pth = '/home/trevor/lipEstimation/alex_save/'

inds = [0,3,6,8,10]
sn_lipest = []
for i, ind in enumerate(inds):
    su = torch.load(os.path.join(pth, 'feat-singular-Conv2d-' + str(ind)))
    sn_lipest.append(su[0].item())

inds = [1,4,6]
for i, ind in enumerate(inds):
    su = torch.load(os.path.join(pth, 'feat-singular-Linear-' + str(ind)))
    sn_lipest.append(su[0].item())
print(sn_lipest)

###############################################################################
# my spectral norms
net, layers = alexnet.alexnet_and_layers()
net = net.to(my_config.device)
x0 = torch.rand(1,3,224,224)
x0 = x0.to(my_config.device)

# iterate over each layer and get the output of each function
n_layers = len(layers)
X = [x0]
for i in range(n_layers):
    f = layers[i]
    X.append(f(X[-1]))

sn_my = []
inds = [0,3,6,8,10, # conv
        16,19,21]   # fully-connected
for ind in inds:
    sn_i, V = utils.conv_power_iteration(layers[ind], X[ind].shape)
    sn_my.append(sn_i.item())
print(sn_my)

diffs = [sn_my[i] - sn_lipest[i] for i in range(len(sn_my))]
print(diffs)
