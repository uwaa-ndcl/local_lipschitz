'''
generate the weight file using this command then run

$ python other_methods/lipsdp_python/solve_sdp.py --form neuron --weight-path data/compnet/lipsdp/weights.mat
'''
import os
import sys
import pathlib
import numpy as np
from scipy.io import savemat
import subprocess
import torch.nn as nn

import utils
import my_config

#import networks.compnet as exp
import networks.tiny as exp
#import networks.mnist as exp
#import networks.cifar10 as exp # memory error

device = my_config.device 

# files
save_dir = os.path.join(exp.main_dir, 'lipsdp/')
weight_file = os.path.join(save_dir, 'weights.mat')

#
net = exp.net()
net = net.to(device)
net.eval()
x0 = exp.x0
x0.requires_grad = False
layers = net.layers
n_layers = len(layers)

# get all inputs and outputs
layer_input = []
layer_output = []
outpt = x0
for layer in layers:
    inpt = outpt
    outpt = layer(inpt)
    layer_input.append(inpt)
    layer_output.append(outpt)

weights = []
biases = []

for i,layer in enumerate(layers):
    # affine before ReLU
    if (i+1<n_layers) and isinstance(layer, nn.Linear) and isinstance(layers[i+1], nn.ReLU):
        weights.append(layers[i].weight.detach().cpu().numpy().astype(np.float64))
        biases.append(layers[i].bias.detach().cpu().numpy().astype(np.float64))
        print('affine')

    # sole affine - must be the last layer for the LipSDP implementation!
    elif isinstance(layer, nn.Linear):
        weights.append(layers[i].weight.detach().cpu().numpy().astype(np.float64))
        biases.append(layers[i].bias.detach().cpu().numpy().astype(np.float64))
        print('sole affine')
        if i != n_layers-1:
            print('ERROR: SOLE AFFINE FUNCTION MUST BE THE LAST LAYER!!!')
        
    # ReLU after affine
    elif (i-1>=0) and isinstance(layer, nn.ReLU) and isinstance(layers[i-1], nn.Linear):
        print('ReLU')

    # conv
    elif isinstance(layer, nn.Conv2d):
        W = utils.conv_matrix(layer, layer_input[i].shape)
        W = W.detach().cpu().numpy().astype(np.float64)
        b = utils.conv_bias_vector(layer, layer_output[i].shape)
        b = b.detach().cpu().numpy().astype(np.float64)
        weights.append(W)
        biases.append(b)
        print('conv2D')

    # ReLU after conv
    elif (i-1>=0) and isinstance(layer, nn.ReLU) and isinstance(layers[i-1], nn.Conv2d):
        print('ReLU')

    # sole ReLU
    elif isinstance(layer, nn.ReLU):
        print('sole ReLU - NOT IMPLEMENTED!!!')

    # flatten
    elif isinstance(layer, nn.Flatten):
        print('flatten')

    # max pool
    elif isinstance(layer, nn.MaxPool2d):
        mp_lip = utils.max_pool_lip(layer)
        print('max pool (lipshitz constant = ', mp_lip, ')', sep='')

    # other
    else:
        print('other layer - NOT IMPLEMENTED!!!')

# list of weight matrices for affine-ReLU sequences
# save matrices to file
arr = np.empty(len(weights), dtype=object)
arr[:] = weights
data = {'weights': arr}
savemat(weight_file, data)
