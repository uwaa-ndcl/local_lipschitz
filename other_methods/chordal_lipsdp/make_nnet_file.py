import torch.nn as nn
from other_methods.chordal_lipsdp import writeNNet

import tiny as exp

net = exp.net()

# iterate through network
layers = net.layers
n_layers = len(layers)
print('LAYERS:')

#
weights = []
biases = []

for i,layer in enumerate(layers):
    # affine before ReLU
    if (i+1<n_layers) and isinstance(layer, nn.Linear) and isinstance(layers[i+1], nn.ReLU):
        weights.append(layers[i].weight)
        biases.append(layers[i].bias)
        print('affine')

    # sole affine
    elif isinstance(layer, nn.Linear):
        print('sole affine - NOT IMPLEMENTED!!!')

    # ReLU after affine
    elif (i-1>=0) and isinstance(layer, nn.ReLU) and isinstance(layers[i-1], nn.Linear):
        print('ReLU')

    # sole ReLU
    elif isinstance(layer, nn.ReLU):
        print('sole ReLU - NOT IMPLEMENTED!!!')

    # conv
    elif isinstance(layer, nn.Conv2d):
        print('conv2D - NOT IMPLEMENTED!!!')

    # other
    else:
        print('other layer - NOT IMPLEMENTED!!!')

# other variables
n_input = weights[0].shape[1]
n_output = weights[-1].shape[0]
inputMins = [0]*n_input # I don't think this is used in this case
inputMaxes = [1]*n_input # I don't think this is used in this case
means = [0]*(n_input+1) # I don't think this is used in this case
ranges = [0]*(n_input+1) # I don't think this is used in this case

# make the file
filename = 'other_methods/chordal_lipsdp/network.nnet'
writeNNet.writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, filename)
