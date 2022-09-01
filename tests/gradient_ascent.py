import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import my_config
import utils
from networks import mnist

# setup
device = my_config.device 
mode = 'mnist'

main_dir = dirs.mnist_dir
net = mnist.mnist_net()
net = net.to(device)
net.eval()

# load image
#filename = os.path.join(main_dir, '2.png')
filename = os.path.join(main_dir, '8.png')
x = Image.open(filename)
x = mnist.transform(x)
x = torch.unsqueeze(x, 0)
x = x.to(device)

layers = mnist.get_layers(net)

# evaluate the nominal input
y = net(x)

# gradient ascent
n_step = 30
step_size = 10**-1
ind = 4 # ind of output to ascend
xc = x
eps_step = np.full(n_step, np.nan)
lb_step = np.full(n_step, np.nan)
for i in range(n_step):
    J = utils.jacobian(net, xc)
    J0 = J[ind,:]
    pert = J0.view(x.shape)
    xc = xc + step_size*pert # "xc += pert" throws an error
    yc = net(xc)
    sm = nn.functional.softmax(yc, dim=1)
    print('pred', sm[0,ind].item())
    eps_step[i] = torch.norm(x - xc)
    lb_step[i] = torch.norm(y - yc)/torch.norm(x - xc)
    print('eps  ', eps_step[i].item())
    print('lb   ', lb_step[i].item())
