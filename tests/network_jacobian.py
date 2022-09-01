'''
get the jacobian of a full network
'''

import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage

import utils as utils
import dirs
import my_config
from networks import mnist, cifar10, alexnet

# setup
device = my_config.device
main_dir = dirs.mnist_dir
net = mnist.mnist_net()
net = net.to(device)
net.eval()

# load image
filename = os.path.join(main_dir, '8.png')
x0 = Image.open(filename)
x0 = mnist.transform(x0)
img = ToPILImage()(x0)
img.save('/home/trevor/Downloads/xtrans.png')
x0 = torch.unsqueeze(x0, 0)
x0 = x0.to(device)
x0_shape = x0.shape
y0 = net(x0)
out0 = nn.functional.softmax(y0, dim=1)
top1 = torch.topk(out0, 1)

# test jacobian for array-to-array function
conv1 = net.conv1
relu = nn.functional.relu
#layer_fun = lambda a : relu(conv1(a))
layer_fun = lambda a : conv1(a)
z0 = layer_fun(x0)
J = utils.jacobian(layer_fun, x0)
J_mat = J.view(J.shape[0], -1)
J_mat_cc = utils.conv_matrix(conv1, x0.shape)
print('J err', torch.norm(J_mat-J_mat_cc))

# get Jacobian
J = utils.jacobian(net, x0, batch_size=10)
J2 = utils.jacobian(net, x0, batch_size=2)
J3 = utils.jacobian(net, x0, batch_size=3)
#print(torch.norm(J2-J3))

# print (for all methods)
eps = 100
print('\033[4mi |  1st   | 2nd    | lip \033[0m')
def print_output(x, y, str):
    lip = torch.norm(y0 - y)/torch.norm(x0 - x)
    out = nn.functional.softmax(y, dim=1)
    top3_val, top3_ind = torch.topk(out, 3) 
    print(str, '|', top3_ind[0][0].item(), '%.2f' % top3_val[0][0].item(), '|',
                    top3_ind[0][1].item(), '%.2f' % top3_val[0][1].item(), '|',
                    '%.2f' % lip.item())

# SVD
J_mat = J.view(J.shape[0], -1)
u, s, v = torch.svd(J_mat)
vv = v[:,0] # max singular vector
grad = vv.view(x0.shape)

# perturb in SVD direction
grad *= torch.norm(grad) # normalize
x = x0 + eps*grad
img = ToPILImage()(x[0,:,:,:])
img.save('/home/trevor/Downloads/xsvdpert.png')
y = net(x)
print_output(x,y,'s')

# network-wide
x = utils.all_class_adv(net, x0, eps)
y = net(x)
print_output(x,y,'n')

# perturb in each class direction
for i in range(10):
    x = utils.FGSM(net, x0, i, eps, normalize=True)
    #x = utils.FGM(net, x0, i, eps, normalize=True)
    y = net(x)
    img = ToPILImage()(x[0,:,:,:])
    img.save('/home/trevor/Downloads/xpert.png')
    print_output(x,y,str(i))

# 1st conv+relu
conv1 = net.conv1
relu = nn.functional.relu
layer_fun = lambda a : relu(conv1(a))
x2 = layer_fun(x0)

#x_pert = utils.all_class_adv(net, x, eps)
utils.lower_bound_adv(layer_fun, x, [1, 2], save_npz='/home/trevor/Downloads/dumb.npz')
