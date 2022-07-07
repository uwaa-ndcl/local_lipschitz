import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks
from PIL import Image

import utils
import my_config

#import tiny as exp 
import mnist as exp

def normalize(x):
    '''
    x = (x-mean)/std
    '''
    return (x-exp.train_mean[0])/exp.train_std[0]

def unnormalize(x):
    '''
    put x back to [0,1] interval
    '''
    return x*exp.train_std[0] + exp.train_mean[0]

# the same network as the original except normalization is done as a network operation
class LeNet01(nn.Module):

    def __init__(self):
        super(LeNet01, self).__init__()
        self.conv1 = net.conv1
        self.conv2 = net.conv2

        self.maxpool = net.maxpool

        self.fc1 = net.fc1
        self.fc2 = net.fc2
        self.fc3 = net.fc3

    def forward(self, x):
        x = (x-exp.train_mean[0])/exp.train_std[0]
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# original network
net = exp.net()
x0_nml = exp.x0
x0_01 = unnormalize(x0_nml)

# network with inputs on interval [0,1]
net_01 = LeNet01()
y0_nml = net(x0_nml)
y0_01 = net_01(x0_01)
print('nml v 01 for y err:', torch.norm(y0_01 - y0_nml))

# utils function
print('\nUTILS FUNCTION')
xa_01, diffs_01 = utils.cw_attack(net_01, x0_01)
xa_nml = normalize(xa_01)
y_nml = net(xa_nml)
x0_nml_vec = torch.flatten(x0_nml, start_dim=1, end_dim=3)
xa_nml_vec = torch.flatten(xa_nml, start_dim=1, end_dim=3)
diff_nrm = torch.norm(xa_nml_vec - x0_nml_vec, dim=1)
max_diff_nrm = torch.max(diff_nrm).item()
print('successful attack input diffs (normalized):', diff_nrm)
print('max successful attack input diff (normalized):', max_diff_nrm)

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
attack = torchattacks.CW(net_01, c=1e0, kappa=0, steps=1000, lr=0.01)
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
    xa_nml = normalize(xa_01)
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
