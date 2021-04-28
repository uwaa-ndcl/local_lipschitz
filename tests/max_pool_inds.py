import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from lip import my_config
import lip.directories as dirs
import lip.network.utils as ut
from lip.network import mnist, cifar10, alexnet

device = my_config.device

# create max pool function
fun = torch.nn.MaxPool2d(kernel_size=4, stride=3, padding=1)
x = torch.rand(1,7,4,3)
y = fun(x)
x_flat = x.flatten()
y_flat = y.flatten()
n_x = x.numel() 
n_y = y.numel() 
print('x shape', x.shape)
print('y shape', y.shape)

# get inds
x_list, y_list = ut.max_pool_inds(fun, x.shape, batch_size=5)
#print(x_list)
#print(y_list)
#print(len(x_list))
#print(len(y_list))
lens = [len(xi) for xi in x_list]
print('max # of times an element of x shows up in the output:', np.max(lens))

# try to apply max pool function using inds from function above
y_flat_test = torch.empty(n_y)
for i in range(n_y):
    y_flat_test[i] = torch.max(x_flat[y_list[i]])
y_test = y_flat_test.view(y.shape)
print('y - y diff', torch.norm(y - y_test))

###############################################################################
# max pool lipschitz
main_dir = dirs.mnist_dir
net = mnist.mnist_net()
net = net.to(device)
net.eval()
layers = mnist.get_layers(net)

# load image
#filename = os.path.join(main_dir, '2.png')
filename = os.path.join(main_dir, '8.png')
x0 = Image.open(filename)
x0 = mnist.transform_test(x0)
x0 = torch.unsqueeze(x0, 0)
x0 = x0.to(device)

# iterate over each layer and get the output of each function
n_layers = len(layers)
X = [x0]
for i in range(n_layers):
    f = layers[i]
    X.append(f(X[-1]))

aff_relu0 = layers[0]
maxpool1 = layers[1]
aff_relu2 = layers[2]
maxpool3 = layers[3]
test1 = maxpool1(X[1])
test3 = maxpool3(X[3])
print(torch.norm(test1 - X[2]))
print(torch.norm(test3 - X[4]))

eps = .1
l0 = ut.get_l(aff_relu0[0], X[0].shape, X[1].shape)
l2 = ut.get_l(aff_relu2[0], X[2].shape, X[3].shape)
b0 = ut.conv_bias_vector(aff_relu0[0], X[1].shape)
b2 = ut.conv_bias_vector(aff_relu2[0], X[3].shape)
sn0, V = ut.conv_power_iteration(aff_relu0[0], X[0].shape)
sn3, V = ut.conv_power_iteration(aff_relu2[0], X[2].shape)
out_maxes0 = b0 + eps*l0
out_maxes2 = b2 + eps*l2
max0 = maxpool1(out_maxes0.view(X[1].shape))
max3 = maxpool3(out_maxes2.view(X[3].shape))
lip0_maxpool = torch.norm(max0)/eps
lip3_maxpool = torch.norm(max3)/eps
lip0_eps = eps*sn0
lip3_eps = eps*sn3

print('lip 0, maxpool technique', lip0_maxpool)
print('lip 3, maxpool technique', lip3_maxpool)
print('lip 0, eps*n_appear technique', lip0_eps)
print('lip 3, eps*n_appear technique', lip3_eps)

###############################################################################
# alexnet
net, layers = alexnet.alexnet_and_layers()

# iterate over each layer and get the output of each function
n_layers = len(layers)
x0 = torch.rand(1,3,224,224) 
X = [x0]
for i in range(n_layers):
    f = layers[i]
    X.append(f(X[-1]))

maxpool2 = layers[2]
x_list, y_list = ut.max_pool_inds(maxpool2, X[2].shape)
print(x_list)
lens = [len(xi) for xi in x_list]
print(maxpool2)
print('max # of times an element of x shows up in the output:', np.max(lens))
