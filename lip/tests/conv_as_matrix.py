import copy
import numpy as np
import torch
import torch.nn as nn
import scipy as sp

from lip import my_config
import lip.network.utils as ut

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
# ORIGINAL
# make matrix and bias vector
# simpler formula
A = ut.conv_matrix(conv, X.shape)
b = ut.conv_bias_vector(conv, Y.shape)
y_lin = A @ x + b

# matrix svd
u,sss,v = torch.svd(A)
spec_norm_svd = torch.max(sss)

# power iteration
n_iter = 99
spec_norm_pow, v = ut.conv_power_iteration(conv, X.shape, n_iter=n_iter)

print('\nWITHOUT RELU')
print('conv v. matrix error:', torch.norm(y_lin - y).item())
print('spec norm svd', spec_norm_svd.item())
print('spec norm pow', spec_norm_pow.item())
print('spec norm error:', torch.norm(spec_norm_svd - spec_norm_pow).item())


###############################################################################
# WITH RELU
# apply relu
m = A.shape[0]
rand_vals = torch.rand(m)
ignores = torch.ones(m)
keep_inds = (y>0).to(torch.bool)
zero_output_inds = ~keep_inds
R = torch.diag(keep_inds.to(torch.float))
RA = R @ A
Rb = R @ b

# matrix representation with relu
y_relu = nn.functional.relu(y)
y_lin_relu = RA @ x + Rb

# svd with relu
u_new, s_new, v_new = torch.svd(RA)
spec_norm_svd = torch.max(s_new)

# power iteration with relu
spec_norm_pow, v = ut.conv_power_iteration(conv, X.shape, zero_output_inds=zero_output_inds, n_iter=100)
#spec_norm, v = ut.conv_power_iteration(conv, X.shape, n_iter=999)

print('\nWITH RELU')
print('conv v. matrix error:', torch.norm(y_lin_relu - y_relu).item())
print('spec norm svd', spec_norm_svd.item())
print('spec norm pow', spec_norm_pow.item())
print('spec norm err', torch.norm(spec_norm_svd - spec_norm_pow).item()) 


###############################################################################
# WITH INPUT AND OUTPUT ZEROS
# input
m = A.shape[0]
n = A.shape[1]

keep_inds_in = (torch.rand(n)>.5).to(torch.bool)
keep_inds_out = (torch.rand(m)>.5).to(torch.bool)
zero_input_inds = ~keep_inds_in
zero_output_inds = ~keep_inds_out
R_in = torch.diag(keep_inds_in.to(torch.float))
R_out = torch.diag(keep_inds_out.to(torch.float))
A_new = R_out @ A @ R_in

# svd with relu
u_new, s_new, v_new = torch.svd(A_new)
spec_norm_svd = torch.max(s_new)

# power iteration with relu
spec_norm_pow, v = ut.conv_power_iteration(conv, X.shape, zero_input_inds=zero_input_inds, zero_output_inds=zero_output_inds, n_iter=200)
#spec_norm, v = ut.conv_power_iteration(conv, X.shape, n_iter=999)

print('\nINPUT AND OUTPUT MASKS')
print('spec norm svd', spec_norm_svd.item())
print('spec norm pow', spec_norm_pow.item())
print('spec norm err', torch.norm(spec_norm_svd - spec_norm_pow).item())


###############################################################################
# WITH INPUT AND OUTPUT ZEROS
m = 34
n = 56
lin = torch.nn.Linear(n,m)
W = lin.weight

keep_inds_in = (torch.rand(n)>.5).to(torch.bool)
keep_inds_out = (torch.rand(m)>.5).to(torch.bool)
zero_input_inds = ~keep_inds_in
zero_output_inds = ~keep_inds_out
R_in = torch.diag(keep_inds_in.to(torch.float))
R_out = torch.diag(keep_inds_out.to(torch.float))
W_new = R_out @ W @ R_in

# svd
u_new, s_new, v_new = torch.svd(W_new)
spec_norm_svd = torch.max(s_new)

# power iteration with relu
spec_norm_pow, v = ut.conv_power_iteration(lin, n, zero_input_inds=zero_input_inds, zero_output_inds=zero_output_inds, n_iter=200)
#spec_norm, v = ut.conv_power_iteration(conv, X.shape, n_iter=999)

print('\nINPUT AND OUTPUT MASKS, LINEAR')
print('spec norm svd', spec_norm_svd.item())
print('spec norm pow', spec_norm_pow.item())
print('spec norm err', torch.norm(spec_norm_svd - spec_norm_pow).item())
