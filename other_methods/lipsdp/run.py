import os
import sys
import pathlib
import numpy as np
from scipy.io import savemat
import subprocess

# add main directory to system path so I can import the files
script_dir = pathlib.Path(__file__).parent.absolute() 
main_dir = script_dir.parent
sys.path.insert(0, main_dir)

import my_config
import utils
import solve_sdp

# files
device = my_config.device 
save_dir = os.path.join(dirs.mnist_dir, 'lipsdp/')
#weight_file = os.path.join(save_dir, 'weights.mat')
weight_file = os.path.join(save_dir, 'random_weights.mat')
#py_file = '/home/trevor/affine_relu_sensitivity/aff_relu/third_party/lipsdp/solve_sdp.py'
solve_sdp_py_file = solve_sdp.__file__

'''
# random weights (documented)
weights = []
net_dims = [2, 10, 30, 20, 2]
num_layers = len(net_dims) - 1
norm_const = 1 / np.sqrt(num_layers)
for i in range(1, len(net_dims)):
  weights.append(norm_const * np.random.rand(net_dims[i], net_dims[i-1]))

# random weights
weight_1 = np.random.rand(5,3)
weight_2 = np.random.rand(4,5)
weight_3 = np.random.rand(2,4)
weights = [weight_1, weight_2, weight_3]
'''
# neural network
import networks.mnist as exp
#import networks.cifar10 as exp # memory error
net = exp.net()
net = net.to(device)
net.eval()
x0 = exp.x0
x0.requires_grad = False

# iterate over each layer and get the output of each function
layers = net.layers
n_layers = len(layers)
X = [x0]
for i in range(n_layers):
    f = layers[i]
    X.append(f(X[-1]))

# weight matrices of convolution and fully-connected layers
weight_mat_1 = ut.conv_matrix(net.conv1, X[0].shape).detach().cpu().numpy().astype(np.float64)
weight_mat_2 = ut.conv_matrix(net.conv2, X[2].shape).detach().cpu().numpy().astype(np.float64)
weight_mat_3 = net.fc1.weight.detach().cpu().numpy().astype(np.float64)
weight_mat_4 = net.fc2.weight.detach().cpu().numpy().astype(np.float64)
weight_mat_5 = net.fc3.weight.detach().cpu().numpy().astype(np.float64)
eye_1 = np.eye(weight_mat_1.shape[0])
eye_2 = np.eye(weight_mat_2.shape[0])

# list of weight matrices for affine-ReLU sequences
# add an identity to single affine-ReLUs,
# see: https://github.com/arobey1/LipSDP/issues/4#
weights_conv1 = [weight_mat_1, eye_1]
weights_conv2 = [weight_mat_2, eye_2]
weights_fc = [weight_mat_3, weight_mat_4, weight_mat_5]
weight_lists = [weights_conv1, weights_conv2, weights_fc]

# get LipSDP estimate of each affine-ReLU sequence
for i, weight_list in enumerate(weight_lists):
    print('affine-ReLU sequence', i)

    # save matrices to file
    arr = np.empty(len(weight_list), dtype=object)
    arr[:] = weight_list
    data = {'weights': arr}
    savemat(weight_file, data)

    # run LipSDP
    cmd = 'python ' + solve_sdp_py_file + ' --form neuron --weight-path ' + weight_file
    subprocess.run([cmd], shell=True)
