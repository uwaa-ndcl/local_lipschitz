'''
test the Fast Gradient Sign Method (FGSM) and Iterative Fast Gradient Sign
Method (IFGSM) adversarial attacks
'''
import torch
import torch.nn.functional as F

import my_config
import utils

#import tiny as exp 
import mnist as exp
#import cifar10 as exp
#import alexnet as exp
#import vgg16 as exp

device = my_config.device 

net = exp.net()
net = net.to(device)
net.eval()
x0 = exp.x0
x0_clone = x0.detach().clone()
x0_clone2 = x0.detach().clone()

# test torch.clip() function
a = torch.randint(0,10,(7,))
#print(a)
b = a.clip(0,5)
#print(b)

# run x0 through network
y0 = net(x0)
ind_true = torch.topk(y0.flatten(), 1)[1].item()
print('ind true:', ind_true)

# test fgsm
#eps = .66 # used in MNIST iteration in other file
eps = 3e-1
ind_new, pert_norm = utils.fgsm(net, x0, eps)
print('\nFGSM')
print('ind new:', ind_new)
if ind_new == ind_true:
    print('perturbation not found')
else:
    print('perturbation norm:', pert_norm)
y0_new = net(x0_clone)
print('y err', torch.norm(y0 - y0_new))

# test ifgsm
eps = 3e-5
#lower = (0 - exp.train_mean[0])/exp.train_std[0]
#upper = (1 + exp.train_mean[0])/exp.train_std[0]
ind_new, pert_norm, n_iters = utils.ifgsm(net, x0, eps, max_steps=10000)
#ind_new, pert_norm, n_iters = utils.ifgsm(net, x0, eps, clip=True, lower=lower, upper=upper)
print('\nI-FGSM')
print('# of iterations:', n_iters)
print('ind new:', ind_new)
if ind_new == ind_true:
    print('perturbation not found')
else:
    print('perturbation norm:', pert_norm)
y0_new2 = net(x0_clone2)
print('y err', torch.norm(y0 - y0_new2))
