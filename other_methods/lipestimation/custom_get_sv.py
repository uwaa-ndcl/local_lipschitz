# compute n_sv highest singular vectors for every convolution
import os
import torch
import torchvision
import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

import my_config

# network
#import networks.compnet as exp
import networks.tiny as exp
#import networks.mnist as exp
#import networks.cifar10 as exp
#import networks.alexnet as exp
#import networks.vgg16 as exp

n_sv = 200
use_cuda = (my_config.device=='cuda')

def spec_net(self, input, output):
    print(self)
    if is_convolution_or_linear(self):
        s, u, v = k_generic_power_method(self.forward, self.input_sizes[0],
                n_sv,
                max_iter=500, use_cuda=use_cuda)
        self.spectral_norm = s
        self.u = u
        self.v = v

    if is_batch_norm(self):
        # one could have also used generic_power_method
        s = lipschitz_bn(self)
        self.spectral_norm = s


def save_singular(net, save_dir):
    # save for all functions
    functions = net.layers
    #functions = net.functions
    for i in range(len(functions)):
        if hasattr(functions[i], 'spectral_norm'):
            torch.save(functions[i].spectral_norm, open(os.path.join(save_dir, 'feat-singular-{}-{}'.format(functions[i].__class__.__name__, i)), 'wb'))
        if hasattr(functions[i], 'u'):
            torch.save(functions[i].u, open(os.path.join(save_dir, 'feat-left-sing-{}-{}'.format(functions[i].__class__.__name__, i)), 'wb'))
            torch.save(functions[i].v, open(os.path.join(save_dir, 'feat-right-sing-{}-{}'.format(functions[i].__class__.__name__, i)), 'wb'))

save_dir = os.path.join(exp.main_dir, 'lipestimation/')

net = exp.net()
net = net.to(my_config.device)
net.eval()

for p in net.parameters():
    p.requires_grad = False

compute_module_input_sizes(net, exp.x0.shape)
execute_through_model(spec_net, net)

save_singular(net, save_dir)
