import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import my_config

net_name = 'tiny'
plot_name = 'Tiny Net'

# this is only for computation and plotting
batch_size_l = 10**4
batch_size_lb = 10**7
batch_size_sn = 100
batch_size_ball = 10**5
eps_min = .1
eps_max = 10
step_size_grad = 1e-4
main_dir = 'data/tiny/'

# input size
n = 3

# set random seed for reproducibility (for both torch.rand and torch.nn.Linear)
torch.manual_seed(0)
A = torch.tensor([[3,-1,2],[-2,7,1],[5,-3,-1]])
b = torch.tensor([3,4,2])

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        #self.fc1 = nn.Linear(n,4)
        #self.fc2 = nn.Linear(4,2)
        #self.fc3 = nn.Linear(5,5)

        
        self.fc1 = nn.Linear(n,n)
        self.fc1.weight.data.copy_(A)
        self.fc1.bias.data.copy_(b)

        self.fc2 = nn.Linear(n,n)
        self.fc2.weight.data.copy_(torch.eye(n))
        self.fc2.bias.data.fill_(0.0)

    def forward(self, x):
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        '''
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# nominal input
x0 = torch.rand(1,n)
#print('x0', x0)
x0 = x0.to(my_config.device)
x0 = x0 + torch.tensor([-1,2,2]).to(my_config.device) # so the output of the first layer won't be all 0

def net():
    net = MyNet()
    # create a list of the network's layers 
    relu = torch.nn.ReLU(inplace=False)
    '''
    net.layers = [net.fc1, relu,
                  net.fc2, relu,
                  net.fc3]
    '''
    net.layers = [net.fc1, relu,
                  net.fc2]

    return net
