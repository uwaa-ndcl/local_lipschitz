import torch

import utils

#import tiny as exp 
import mnist as exp

net = exp.net()
x0 = exp.x0

# test torch.clip() function
a = torch.randint(0,10,(7,))
print(a)
b = a.clip(0,5)
print(b)

# run x0 through network
y0 = net(x0)
ind_true = torch.topk(y0.flatten(), 1)[1].item()
print('ind true:', ind_true)

# test fgsm
eps = .66 # used in MNIST iteration in other file
ind_new, pert_norm = utils.fgsm_new(net, x0, eps)
print('\nFGSM')
print('ind new:', ind_new)
print('perturbation norm:', pert_norm)

# test ifgsm
eps = .0001
lower = (0 - exp.train_mean[0])/exp.train_std[0]
upper = (1 + exp.train_mean[0])/exp.train_std[0]
ind_new, pert_norm, n_iters = utils.ifgsm(net, x0, eps)
#ind_new, pert_norm, n_iters = utils.ifgsm(net, x0, eps, clip=True, lower=lower, upper=upper)
print('\nI-FGSM')
print('# of iterations:', n_iters)
print('ind new:', ind_new)
print('perturbation norm:', pert_norm)
