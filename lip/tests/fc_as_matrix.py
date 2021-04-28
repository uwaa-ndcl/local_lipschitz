# this script shows how a fully-connected layer can be represented as an affine
# transformation (which is obvious), and tests the power iteration method
import torch
from torchvision import models

from lip import my_config
import lip.network.utils as ut

device = my_config.device
if device == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# affine representation
###############################################################################
net = models.alexnet(pretrained=True)
#fc = net.classifier[1]
#fc = net.classifier[4]
fc = net.classifier[6]

A = fc.weight.data
b = fc.bias.data

m,n = A.shape
x0 = torch.rand(n)
x1_aff = A @ x0 + b
x1_real = fc(x0)
print('affine representation error', torch.norm(x1_aff - x1_real).item())

# power iteration
###############################################################################
#sn, v = ut.conv_power_iteration(fc, x0.shape)
u,s,v = torch.svd(A)
sn_true = s[0]
sn_pow, v = ut.conv_power_iteration(fc, x0.shape, n_iter=100)
print('spec norm true', sn_true.item())
print('spec norm pow', sn_pow.item())
print('spec norm error', torch.norm(sn_true - sn_pow).item())

# spec norm with ignores
ignore_inds_1 = (torch.rand(m)>.7)
ignore_inds_2 = (torch.rand(m)>.2)
ignore_inds = torch.stack((ignore_inds_1, ignore_inds_2), dim=0)
R_1 = torch.diag((~ignore_inds_1).to(torch.float))
R_2 = torch.diag((~ignore_inds_2).to(torch.float))
RA_1 = R_1 @ A
RA_2 = R_2 @ A
RA = torch.stack((RA_1,RA_2), dim=0)

u,s,v = torch.svd(RA)
sn_true = s[:,0]
sn_pow, v = ut.conv_power_iteration(fc, x0.shape, ignore_inds=ignore_inds, n_iter=100)
print('spec norm true', sn_true)
print('spec norm pow', sn_pow)
print('spec norm error', torch.norm(sn_true - sn_pow).item())
