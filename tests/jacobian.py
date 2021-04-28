import torch
import torch.nn as nn
import numpy as np

from lip import my_config
import lip.network.utils as ut

###############################################################################
# test getting Jacobian by rows
n0 = 7
n1 = 13
n2 = 5
lyr0 = nn.Linear(n0,n1)
lyr1 = nn.Linear(n1,n2)
fun = nn.Sequential(lyr0, lyr1)

x0 = torch.rand(1,n0)
y0 = fun(x0)
J = ut.jacobian_old(fun, x0, batch_size=10, row_inds=None)

J1 = ut.jacobian_old(fun, x0, batch_size=10, row_inds=[0,2])
J2 = ut.jacobian_old(fun, x0, batch_size=10, row_inds=[2,3])
J3 = ut.jacobian_old(fun, x0, batch_size=10, row_inds=[3,5])

J_new = torch.cat((J1, J2, J3))
print(J)
print('diff', torch.norm(J - J_new))

#eye = torch.eye(y0.shape[1])[[0],:]
#J_i = torch.autograd.grad(outputs=[y0], inputs=[x0], grad_outputs=[eye], retain_graph=True)[0]

###############################################################################
# test Jacobian-times-matrix iteration
K = ut.jacobian(fun, x0, batch_size=10, row_inds=None)

K1 = ut.jacobian(fun, x0, batch_size=10, row_inds=[0,2])
K2 = ut.jacobian(fun, x0, batch_size=10, row_inds=[2,3])
K3 = ut.jacobian(fun, x0, batch_size=10, row_inds=[3,5])

K_new = torch.cat((K1, K2, K3))

###############################################################################
# concat
'''
x0_0 = torch.rand(1,4)
x0_0.requires_grad = True
x0_1 = torch.rand(1,3)
x0_1.requires_grad = False
x0n = torch.cat((x0_0, x0_1), dim=1)
y0n = fun(x0n)
'''
print('jacobian v. jacobian2 diff', torch.norm(J_new - K_new))

#J_in = torch.autograd.grad(outputs=[y0n], inputs=[x0n], grad_outputs=[eye], retain_graph=True)[0]

###############################################################################
# jacobian cols
col_inds = [3,6]
Jc = ut.jacobian_col(fun, x0, col_inds=col_inds, batch_size=1)
print(Jc)

###############################################################################
# jacobian left product

z = torch.rand(5)
J = ut.jacobian(fun, x0)
lp_true = z @ J

lp = ut.jacobian_left_product(fun, x0, z, batch_size=2)
print('left product error', torch.norm(lp - lp_true))
