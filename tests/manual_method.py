'''
manual method to compute local Lipschitz bounds, based on the algorithm in the
paper
this only works for networks which only have fully-connected layers
'''
import torch
import torch.nn as nn

import utils
import networks.compnet as exp

relu = lambda x: (x>0)*x

net = exp.net()
net = net.eval()
layers = net.layers

n_layers = len(layers)
A_list = [None]*n_layers # list of A matrices
b_list = [None]*n_layers # list of b vectors
for i,layer in enumerate(layers):
    if isinstance(layer,nn.Linear): 
        A_list[i] = layer.weight.detach()
        b_list[i] = layer.bias.detach()

x0 = exp.x0.T 
eps = 1e-1
n_input = net.layers[0].in_features
D = torch.eye(n_input)
L_net = 1
k = 0
while k<n_layers:
    layer = layers[k]

    # affine-ReLU
    if (k+1<n_layers) and isinstance(layer, nn.Linear) and isinstance(layers[k+1], nn.ReLU):
        A = A_list[k]
        b = b_list[k]
        b = torch.unsqueeze(b,1)
        y0 = A @ x0 + b 
        m = len(y0)

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
        
        # for next layer
        x0 = relu(A @ x0 + b)
        d = (torch.squeeze(yb)>0).float()
        D = torch.diag(d)
        k += 2
    
    # affine
    elif isinstance(layer, nn.Linear):
        A = A_list[k]
        b = b_list[k]
        b = torch.unsqueeze(b,1)
        y0 = A @ x0 + b 
        m = len(y0)

        #L = torch.linalg.matrix_norm(A, ord=2)
        L = torch.linalg.matrix_norm(A @ D, ord=2)
        #A_norm, V = utils.get_RAD(layer, [], d=None, r_squared=None, n_iter=100)
        A_norm, V = utils.get_RAD(layer, [], d=torch.diag(D), r_squared=None, n_iter=100)
        L2 = A_norm.item()
        print('L2',L2)

        # for next layer
        x0 = relu(A @ x0 + b)
        D = torch.eye(m)
        k += 1

    else:
        print('ERROR: THIS ALGORITHM IS NOT IMPLEMENTED FOR THIS TYPE OF LAYER') 

    eps *= L
    print('L', L)
    L_net *= L

print('local Lipschitz constant is', L_net)
