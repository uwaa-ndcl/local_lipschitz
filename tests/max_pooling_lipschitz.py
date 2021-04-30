'''
Theorem 3 & Proposition 5

this script is a brute-force verification that the max pooling local Lipschitz
result is true
'''
import numpy as np
import torch
import torch.nn as nn

mode = 'paper' # example in figure from paper
mode = 'small' # small example that can be more densely sampled

# x and y dimensions of input array
if mode=='paper':
    n_x = 7
    n_y = 7
elif mode=='small':
    n_x = 4
    n_y = 4
n_inputs = n_x*n_y
n_batch = 10**6 # number of sample vectors

# create max pooling function
if mode=='paper':
    k = 3
    s = 2
elif mode=='small':
    k = 2
    s = 1
pool = nn.MaxPool2d(k, stride=s)

# analytical solution
n_max = np.ceil(k/s)**2
lip_anl = np.sqrt(n_max)
print('analytical:', lip_anl)

# generate some random inputs
X1 = torch.rand(n_batch,n_x,n_y)
X2 = torch.rand(n_batch,n_x,n_y)

# values I know will achieve the max lipschitz
custom_values = 1
if custom_values:
    X1[-1,:,:] = torch.zeros(n_x, n_y) 
    X2[-1,:,:] = torch.zeros(n_x, n_y) 
    if mode=='paper':
        X1[:,4,4] = 1
        X2[:,4,4] = .5
    elif mode=='small':
        X1[:,1,1] = 1
        X2[:,1,1] = .5

# apply max poolng function
Y1 = pool(X1)
Y2 = pool(X2)

# flatten inputs and outputs into vectors
X1 = torch.flatten(X1, start_dim=1)
X2 = torch.flatten(X2, start_dim=1)
Y1 = torch.flatten(Y1, start_dim=1)
Y2 = torch.flatten(Y2, start_dim=1)

# brute-force
lip_num = torch.norm(Y2 - Y1, dim=1)
lip_den = torch.norm(X2 - X1, dim=1)
lip_frac = lip_num/lip_den
lip_frac_max = torch.max(lip_frac).item()
print('brute-force:', lip_frac_max)
