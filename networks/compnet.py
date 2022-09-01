import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import network_bound
import my_config

net_name = 'compnet'
plot_name = 'Comparison Net'

# this is only for computation and plotting
batch_size_l = 10**4
batch_size_lb = 10**7
batch_size_sn = 100
batch_size_ball = 10**5
batch_size_rand = 10**4
eps_min = .5*1e-3
eps_max = 100*1e-3
step_size_grad = 1e-6
main_dir = 'data/compnet/'

# input size
#n_input = 3
n_input = 7
n_output = 10

# set random seed for reproducibility (for both torch.rand and torch.nn.Linear)
torch.manual_seed(0)

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        #self.fc1 = nn.Linear(n_input,30,bias=True)
        #self.fc2 = nn.Linear(30,50,bias=True)
        #self.fc3 = nn.Linear(50,20,bias=True)
        #self.fc4 = nn.Linear(20,10,bias=True)
        self.fc1 = nn.Linear(n_input,20,bias=True)
        self.fc2 = nn.Linear(20,30,bias=True)
        self.fc3 = nn.Linear(30,n_output,bias=True)
        #self.fc4 = nn.Linear(20,10,bias=True)

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc4(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# nominal input
#x0 = torch.rand(1,n_input)
x0 = torch.zeros(1,n_input)
x0 = x0.to(my_config.device)

def net():
    net = MyNet()

    relu = torch.nn.ReLU(inplace=False)
    #net.layers = [net.fc1, relu,
                  #net.fc2, relu,
                  #net.fc3, relu,
                  #net.fc4]
    net.layers = [net.fc1, relu,
                  net.fc2, relu,
                  net.fc3]

    return net

if __name__ == '__main__':
    network = net()
    network = network.to(my_config.device)
    network = network.eval()
    eps = 10**-1
    batch_size = 10**3

    layer_bounds = network_bound.global_bound(network, x0)
    glob_bound = np.prod(layer_bounds)
    local_bound = network_bound.local_bound(network, x0, eps, batch_size=batch_size)
    eps_arr = np.array([eps/10, eps])
    grad_bound = utils.lower_bound_asc(network, x0, eps_arr, step_size=1e-1)

    
    # I-FGSM
    y0 = network(x0).detach()
    criterion = nn.MSELoss()
    #criterion = nn.NLLLoss()
    #x_new = x0.detach().clone()
    #x_new = x0
    #x_new = torch.ones_like(x0)
    x_new = torch.autograd.Variable(torch.ones_like(x0), requires_grad=True)
    #x_new.requires_grad = True
    #x_new.requires_grad = True
    y_new = network(x_new)

    # go until the classification changes
    eps = 1e-2
    i = 0
    lip = -1 # max Lipschitz found
    while i < 1e4:
        # loss
        y_new = network(x_new)
        loss = criterion(y0, y_new)
        network.zero_grad()
        loss.backward()

        # get gradient and create perturbation
        grad = x_new.grad.data
        pert = eps*torch.sign(grad)

        # update input x
        x_new = x_new + pert
        x_new = torch.autograd.Variable(x_new.data, requires_grad=True)

        # lip
        lip_i = torch.linalg.vector_norm(y_new - y0)/torch.linalg.vector_norm(x_new - x0)
        if lip_i > lip:
            lip = lip_i

        i += 1
    ifgsm_bound = lip

    print('upper bound, global:', glob_bound)
    print('upper bound, local:', local_bound)
    print('lower bound, gradient:', grad_bound)
    print('lower bound, ifgsm:', ifgsm_bound)

    '''
    network = net()
    network = network.eval()
    #eps = 10**-3
    eps = 10**-2
    batch_size = 10**3

    torch.set_default_dtype(torch.float64)
    network = network.double()
    x0 = x0.double()

    layer_bounds = network_bound.global_bound(network, x0)
    glob_bound = np.prod(layer_bounds)
    local_bound = network_bound.local_bound(network, x0, eps, batch_size=batch_size)
    #fgsm_bound = 

    #eps = np.linspace(-.1,-.2,10)
    eps_arr = np.array([eps/10, eps])
    x0.requires_grad = True
    grad_bound = utils.lower_bound_asc(network, x0, eps_arr, step_size=1e-6)
    print('upper bound, global:', glob_bound)
    print('upper bound, local:', local_bound)
    print('lower bound, gradient:', grad_bound)

    # test
    xc = torch.tensor([1.0332e-5, 4.8040e-5, -2.3694e-6])
    lip = torch.norm(network(x0)-network(xc))/torch.norm(x0-xc)
    print('lip', lip)

    step_size = 1e-6
    y0 = network(x0)
    J = utils.jacobian(network, x0)
    J0 = J[0,:]
    pert = J0.view(x0.shape)
    xc = x0 + step_size*pert # "xc += pert" throws an error
    yc = network(xc)
    eps_step = torch.linalg.vector_norm(x0 - xc)
    jump_bound = torch.linalg.vector_norm(y0 - yc)/torch.linalg.vector_norm(x0 - xc)
    print('low bound, jump:', jump_bound)

    net_simp = nn.Sequential(network.fc1, nn.ReLU())
    z0 = net_simp(x0)
    yc = network.fc1(xc).T
    zc = net_simp(xc)
    lip1 = torch.linalg.vector_norm(z0-zc)/torch.linalg.vector_norm(x0-xc)
    eps1 = torch.linalg.vector_norm(x0-xc)
    A = network.fc1.weight.detach()
    b = network.fc1.bias.detach()
    b = torch.unsqueeze(b,1)
    y0 = A @ x0.T + b 
    m = len(y0)
    D = torch.eye(3)
    relu = lambda x: (x>0)*x

    yb = torch.zeros(m,1)
    for i in range(m):
        aiT = A[i,:]
        aiT = torch.unsqueeze(aiT,0)
        yb[i] = eps*torch.linalg.vector_norm(aiT @ D).item() + y0[i].item()

    r = torch.zeros(m)
    for i in range(m):
        if yb[i]!=y0[i]:
            r[i] = (relu(yb[i]) - relu(y0[i]))/(yb[i]-y0[i])
    R = torch.diag(r)
    L = torch.linalg.matrix_norm(R @ A @ D, ord=2)
    print('eps example', eps1)
    print('simp net bound example', lip1)
    print('simp net bound eq', L)

    lhs = torch.abs(relu(A@xc.T + b) - relu(A@x0.T + b))
    rhs = R @ torch.abs(A@xc.T + b - (A@x0.T + b))
    '''
