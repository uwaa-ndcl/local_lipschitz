import os
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.io import savemat
import torch
import torch.nn as nn

import my_config
import mnist, cifar10, alexnet
import utils
import solve_sdp

# setup
solve_sdp_file = solve_sdp.__file__ 
lipsdp_dir = str(Path(solve_sdp_file).parent) 

'''
MNIST net layer sizes
784  (1x28x28) conv
3456 (6x24x24) maxpool
864  (6x12x12) conv
1024 (16x8x8)  maxpool
256  (16x4x4)  flatten
256            FC
120            FC
84             FC
10
'''

device = my_config.device 

net = mnist.mnist_net()
net = net.to(device)
net.eval()

# load image
main_dir = exp.main_dir
filename = os.path.join(main_dir, '8.png')
x0 = Image.open(filename)
x0 = mnist.transform_test(x0)
x0 = torch.unsqueeze(x0, 0)
x0 = x0.to(device)
x0.requires_grad= False

# get weight matrices
weights = [] # list of weights as numpy arrays
n_layers = len(net.layers)
X = [x0]
for i, layer in enumerate(net.layers):
    f = net.layers[i]
    X.append(f(X[-1]))
    if isinstance(layer, nn.Sequential):
        if isinstance(layer[0], nn.Conv2d):
            weight_i = utils.conv_matrix(layer[0], X[i].shape)
            weight_i = weight_i.detach().cpu().numpy().astype('float64')

        elif isinstance(layer[0], nn.Linear):
            weight_i = layer[0].weight.detach().cpu().numpy().astype('float64')

        weights.append(weight_i)

# original example
weights = []
net_dims = [2, 1000, 3000, 20, 2]
net_dims = [2, 10]
num_layers = len(net_dims) - 1
norm_const = 1 / np.sqrt(num_layers)
for i in range(1, len(net_dims)):
  weights.append(norm_const * np.random.rand(net_dims[i], net_dims[i-1]))

# run the solve solve_sdp.py file from the shell
def run_lipsdp_layer(weight_path, form='network'):
    '''
    weight_file_path: path to .mat weight file
    form: 'network', 'neuron', or 'layer'
    '''

    my_cmd = \
        'cd ' + lipsdp_dir + '; ' \
        'python ' + solve_sdp_file + ' --form ' + form + ' --weight-path ' + weight_path + '; '\
        'cd -'
    subprocess.run([my_cmd], shell=True)

# LipSDP can't handle the whole list of weights because there are other functions such as max pooling in between. Also, LipSDP requires at least two weight matrices to be input, and the only sequence in MNIST of more than two affine-ReLU layers without a max pool between them is weights[2] and weights[3]. I don't think LipSDP can deal with the final fully-connected layer, which does not have a ReLU.
#weights = weights[2:3]
#weights = [weights[0]]
fname = os.path.join(lipsdp_dir, 'mnist_lipsdp.mat')
data = {'weights': np.array(weights, dtype=np.object)}
savemat(fname, data)

# last two affine-relu layers in mnist:
# network (2149 sec = 36 min), neuron (18 sec), layer (5 sec)
try:
    #run_lipsdp_layer(fname, form='neuron --split --split-size 1')
    run_lipsdp_layer(fname, form='neuron')
except:
    print('LipSDP failed!!!')
