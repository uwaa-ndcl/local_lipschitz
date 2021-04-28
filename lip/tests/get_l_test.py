import copy
import torch
import torch.nn as nn
from torchvision import models, transforms

from lip import my_config
import lip.network.utils as ut

# alexnet
device = my_config.device
alexnet = models.alexnet(pretrained=True)
alexnet = alexnet.eval() # does this do anything?
alexnet.to(device)
conv = alexnet.features[0]

# conv 1
kernel_size0 = conv.kernel_size
weight0 = conv.weight
K = weight0.data.cpu().numpy()
bias0 = conv.bias
s = conv.stride[0] # assume both stride values are the same
p = conv.padding[0] # assume both padding values are the same

# load image
x0 = torch.rand(1,3,67,67).to(device) # random image instead
x0 = x0.to(device)
x1 = conv(x0)
n = x0.numel()
m = x1.numel()
x0_shape = x0.shape
x1_shape = x1.shape

# determine A and calculate linear x1
A = ut.conv_matrix(conv, x0.shape)
conv_nobias = copy.deepcopy(conv)
conv_nobias.bias = None
x0_flat = x0.view(-1,1)
x1_nobias = conv_nobias(x0)
x1_nobias_flat = x1_nobias.view(-1,1)
x1_Ax = A @ x0_flat
print('Ax error ', torch.norm(x1_nobias_flat - x1_Ax).item())

# compare l methods
diag = torch.diag(A@A.T)
l_old = torch.sqrt(diag)
l_new = ut.get_l(conv, x0_shape, x1_shape)
print('l error ', torch.norm(l_old - l_new).item())
