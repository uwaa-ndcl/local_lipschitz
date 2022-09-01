'''
test the Carlini & Wagner adversarial attack
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchattacks

import my_config
import utils

# setup
device = my_config.device 

#import networks.tiny as exp 
import networks.mnist as exp
#import networks.cifar10 as exp
#import networks.alexnet as exp
#import networks.vgg16 as exp

# first things
x0 = exp.x0
x0 = x0.to(device)
net = exp.net()
net = net.to(device)
net.eval()
y0 = net(x0)
class0 = torch.argmax(y0).item()

# attack
x_adv = utils.cw_attack(exp,LEARNING_RATE=1e-3,CONFIDENCE=0,PRINT_PROG=False)

# analyze the results
y_adv = net(x_adv)
class_adv = torch.argmax(y_adv).item()
pert_size = torch.norm(x0-x_adv).item()
print('original class', class0)
print('new class', class_adv)
print('pert size', pert_size) 
