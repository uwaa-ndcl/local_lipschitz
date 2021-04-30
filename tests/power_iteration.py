'''
this script checks if my power iteration to compute the norm of R@A@D is correct
'''
import sys
import copy
import pathlib
import numpy as np
import torch
import torch.nn as nn
import scipy as sp

# add main directory to search path so I can import my other files
tests_dir = pathlib.Path(__file__).parent.absolute()
main_dir = tests_dir.parent.absolute()
sys.path.insert(1, main_dir)

import my_config
import utils

###############################################################################
# SETUP
device = my_config.device
if device == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# setup
# m3_1: Conv2d(153, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
ch_in = 3 # channels
k = 5  # kernel size
r = 31 # rows
c = 31 # cols
ch_out = 5  # filters (output channels)
s = 3  # stride
p = 2  # padding

# torch reshapes last dimenson fastest!
X = torch.rand(1,ch_in,r,c)
K = (torch.rand(ch_out,ch_in,k,k)-.5)*4
#Y = nn.functional.conv2d(X, K, stride=s, padding=p) # output: batch, out chan, H, W
bias = torch.rand(ch_out)

# make torch function
conv = nn.Conv2d(ch_in, ch_out, k, stride=s, padding=p)
conv.weight = torch.nn.Parameter(K)
conv.bias = torch.nn.Parameter(bias)
conv.to(device)
Y = conv(X)
x = X.flatten()
y = Y.flatten()
conv_new = copy.deepcopy(conv)
conv_nobias = copy.deepcopy(conv) 
conv_nobias.bias = None

###############################################################################
# get A matrix corresponding to convolution function
# make matrix and bias vector
A = utils.conv_matrix(conv, X.shape)
b = utils.conv_bias_vector(conv, Y.shape)
m,n = A.shape
x0 = torch.rand(1,ch_in,r,c)
y0 = conv(x0)
x0_flat = torch.flatten(x0)
y0_flat = A @ x0_flat + b
mat_err = torch.norm(torch.flatten(y0) - y0_flat)
mat_err = mat_err.item()
print('conv v. matrix error for random input', mat_err)

###############################################################################
# norm of A

# matrix svd
u,ss,v = torch.svd(A)
norm_svd = torch.max(ss).item()

# power iteration
n_iter = 100
norm_pow, v = utils.get_RAD(conv, X.shape, d=None, r_squared=None, n_iter=n_iter)
norm_pow = norm_pow.item()

print('\nNORM OF A')
print('matrix', norm_svd)
print('pow', norm_pow)
print('matrix v. pow error:', np.abs(norm_svd - norm_pow))


###############################################################################
# norm of R@A
r = torch.rand(m)
r_squared = r**2
R = torch.diag(r)
RA = R@A

# svd
u, ss, v = torch.svd(RA)
norm_svd = torch.max(ss).item()

# power iteration
norm_pow, v = utils.get_RAD(conv, X.shape, d=None, r_squared=r_squared, n_iter=n_iter)
norm_pow = norm_pow.item()

print('\nNORM OF R@A')
print('matrix:', norm_svd)
print('pow:', norm_pow)
print('matrix v. pow error:', np.abs(norm_svd - norm_pow)) 


###############################################################################
# norm of A@D
d = torch.randint(0,2,(n,)).to(torch.float)
D = torch.diag(d)
AD = A@D

# svd
u, ss, v = torch.svd(AD)
norm_svd = torch.max(ss).item()

# power iteration with relu
norm_pow, v = utils.get_RAD(conv, X.shape, d=d, r_squared=None, n_iter=n_iter)
norm_pow = norm_pow.item()

print('\nNORM OF A@D')
print('matrix', norm_svd)
print('pow', norm_pow)
print('matrix v. pow error', np.abs(norm_svd - norm_pow))


###############################################################################
# norm of R@A@D
RAD = R@A@D

# svd
u, ss, v = torch.svd(RAD)
norm_svd = torch.max(ss).item()

# power iteration with relu
norm_pow, v = utils.get_RAD(conv, X.shape, d=d, r_squared=r_squared, n_iter=n_iter)
norm_pow = norm_pow.item()

print('\nNORM OF R@A@D')
print('matrix', norm_svd)
print('pow', norm_pow)
print('matrix v. pow error', np.abs(norm_svd - norm_pow))
