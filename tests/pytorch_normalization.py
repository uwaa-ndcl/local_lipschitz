# this scipt shows that torchvisions' transforms.Normalize() function is an
# affine operator

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

mean = (3,2,1)
std = (7,3,6)

transform = transforms.Compose([
    transforms.Normalize(mean, std),
])

# apply transform
n = 1
ch = 3
h = 300
w = 300
X = torch.rand(ch,h,w)
Y = transform(X)

# subtract mean and divide by standard deviation
mean = torch.FloatTensor(mean)
mean = mean[:,None,None]
std = torch.FloatTensor(std)
std = std[:,None,None]
Y2 = (X-mean)/std
err2 = torch.norm(Y-Y2)
print('function form error', err2.item())

# put into Ax+b form
A = 1/std
B = -mean/std
Y3 = A*X + B
err3 = torch.norm(Y-Y3)
print('affine form error:', err3.item())
