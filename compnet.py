import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import network_bound
import my_config

net_name = 'compnet'
plot_name = 'Comparison Net'

# this is only for computation and plotting
batch_size_l = 10**4
batch_size_lb = 10**7
batch_size_sn = 100
batch_size_ball = 10**5
eps_min = .1
eps_max = 10
step_size_grad = 1e-4
main_dir = 'data/tiny/'

# input size
n_input = 3

# set random seed for reproducibility (for both torch.rand and torch.nn.Linear)
torch.manual_seed(0)

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        self.fc1 = nn.Linear(n_input,30,bias=True)
        self.fc2 = nn.Linear(30,50,bias=True)
        self.fc3 = nn.Linear(50,20,bias=True)
        self.fc4 = nn.Linear(20,10,bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

# nominal input
#x0 = torch.rand(1,n_input)
x0 = torch.zeros(1,n_input)
x0 = x0.to(my_config.device)

def net():
    net = MyNet()

    relu = torch.nn.ReLU(inplace=False)
    net.layers = [net.fc1, relu,
                  net.fc2, relu,
                  net.fc3, relu,
                  net.fc4]

    return net

if __name__ == '__main__':
    import sys
    import pdb; pdb.set_trace()
    current_mod = sys.modules[__name__]
    network = net()
    eps = 10**-3
    batch_size = 10**3
    layer_bounds = network_bound.global_bound(network, x0)
    glob_bound = np.prod(layer_bounds)
    local_bound = network_bound.local_bound(network, x0, eps, batch_size=batch_size)

    x_adv = utils.cw_attack(current_mod, LEARNING_RATE=1e-3, CONFIDENCE=0)
    print('global Lipschitz bound:', glob_bound)
    print('local Lipschitz bound:', local_bound)
