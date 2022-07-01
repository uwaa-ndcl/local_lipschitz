import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks
from PIL import Image
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

# original network
net = exp.net()
x0_nml = exp.x0
x0_01 = unnormalize(x0_nml)

# the same network as the original except normalization is done as a network operation
class LeNetCW(nn.Module):

    def __init__(self):
        super(LeNetCW, self).__init__()
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

net_cw = LeNetCW()

# check if the networks produce equivalent outputs
y0_orig = net(x0_nml)
y0_new = net_cw(x0_01)
print('y diff:', torch.norm(y0_new - y0_orig))

# true index
class_true = torch.topk(y0_orig.flatten(), 1)[1].item()
n = 10
labels = torch.linspace(0,n-1,n,dtype=torch.int64)
#n = 1
#labels = torch.randint(0,10,(1,))
#labels[0] = 8
print('labels:', labels)

# run the attack
attack = torchattacks.CW(net_cw, c=1e-0, kappa=0, steps=1000, lr=0.01)
attack_images = attack(x0_01, labels)
Y = net(attack_images)
attack_classes = torch.topk(Y, 1)[1].flatten().tolist()
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
    xa_01 = attack_images[success_ind,:,:,:]
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
