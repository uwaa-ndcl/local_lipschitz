import torch
import torch.nn as nn
import utils

def relu(x):
    return (x>0)*x

def network_bound(net, x0, eps, batch_size=32):
    '''
    calculate the local Lipschitz bound of a feedforward neural network
    '''

    # get nominal output of each layer
    layers = net.layers
    n_layers = len(layers)
    X0 = [x0]
    for i in range(n_layers):
        f = layers[i]
        X0.append(f(X0[-1]))

    # a list for each layer where each list item is the diagonal elements of
    # the D matrix in other words
    d = [None]*(n_layers+1)

    # bound of network
    L_net = 1 # Lipschitz bound of full network
    #for j, layer in enumerate(layers):
    i = 0
    while i < n_layers:
        layer = layers[i]

        # affine-ReLU
        if i+1<n_layers and isinstance(layer, (nn.Conv2d, nn.Linear)) and isinstance(layers[i+1], nn.ReLU):
            # get l vector
            aiTD = utils.get_aiTD(
                layer, X0[i].shape, X0[i+1].shape,
                d=d[i], batch_size=batch_size)

            # get spectral norm
            with torch.no_grad():
                y0 = layer(X0[i]).flatten() # A@x0 + b
        
            # ybar and R
            y0 = y0.double()
            ybar = eps*aiTD + y0

            # "flat" inds occur when a_i^T D equals zero, and all y_i equal y_{0,i}
            inds_flat = (ybar==y0)
            #if torch.any(inds_flat):
                #print(torch.sum(inds_flat).item(), '/', len(inds_flat), 'flat inds')

            r = (relu(ybar) - relu(y0))/(ybar - y0)
            # this should replace any nans from the prevoius operation with 0s
            r[inds_flat] = 0 
            RAD_norm, V = utils.get_RAD(
                layer, X0[i].shape, d=d[i], r_squared=r**2)
            L = RAD_norm.item()
            #d[i] = (ybar<=0)

            i += 1 # increment i so we skip the ReLU
            d[i+1] = (ybar>0)

        # sole affine
        elif isinstance(layer, (nn.Conv2d, nn.Linear)):
            A_norm, V = utils.get_RAD(layer, X0[i].shape, d=d[i])
            L = A_norm.item()
            d[i+1] = None

        # max pooling
        elif isinstance(layer, nn.MaxPool2d):
            di = d[i].view(X0[i].shape)
            #di = torch.logical_not(di) # the old way
            di = di.to(torch.float)
            d_i1 = layer(di).flatten().to(torch.bool)
            #d_i1 = torch.logical_not(d_i1) # the old way
            d[i+1] = d_i1
            L = utils.max_pool_lip(layer)

        # adaptive average pooling
        # (doesn't change the input when input is nominal sizes)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            # if the adaptive avg pool function doesn't change the input
            if torch.equal(X0[i], X0[i+1]):
                d[i+1] = d[i]
                L = 1
            else:
                print('THE ADAPTIVE AVG POOL SECTION IS NOT IMPLEMENTED FOR INPUTS & OUTPUTS OF DIFFERENT SIZES') 

        # flatten
        elif isinstance(layer, nn.Flatten):
            d_i = layer(torch.unsqueeze(d[i], dim=0))
            d[i+1] = d_i.flatten()
            L = 1

        # dropout
        elif isinstance(layer, nn.Dropout):
            d[i+1] = d[i]
            L = 1

        # any other type of layer
        else:
            print('ERROR: NETWORK BOUND IS NOT IMPLEMENTED FOR THIS TYPE OF LAYER')

        # update
        eps *= L
        L_net *= L
        i += 1

    return L_net
