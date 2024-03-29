import torch.nn as nn

from other_methods.chordal_lipsdp import writeNNet

import utils

import networks.tiny as exp
#import networks.compnet as exp
#import networks.mnist as exp

net = exp.net()
net = net.eval()
x0 = exp.x0

# iterate through network
layers = net.layers
n_layers = len(layers)
print('LAYERS:')

# get all inputs and outputs
layer_input = []
layer_output = []
outpt = x0
for layer in layers:
    inpt = outpt
    outpt = layer(inpt)
    layer_input.append(inpt)
    layer_output.append(outpt)

#
weights = []
biases = []

for i,layer in enumerate(layers):
    # affine before ReLU
    if (i+1<n_layers) and isinstance(layer, nn.Linear) and isinstance(layers[i+1], nn.ReLU):
        weights.append(layers[i].weight)
        biases.append(layers[i].bias)
        print('affine')

    # sole affine - must be the last layer for the LipSDP implementation!
    elif isinstance(layer, nn.Linear):
        weights.append(layers[i].weight)
        biases.append(layers[i].bias)
        print('sole affine')
        if i != n_layers-1:
            print('ERROR: SOLE AFFINE FUNCTION MUST BE THE LAST LAYER!!!')
        

    # ReLU after affine
    elif (i-1>=0) and isinstance(layer, nn.ReLU) and isinstance(layers[i-1], nn.Linear):
        print('ReLU')

    # conv
    elif isinstance(layer, nn.Conv2d):
        W = utils.conv_matrix(layer, layer_input[i].shape)
        b = utils.conv_bias_vector(layer, layer_output[i].shape)
        weights.append(W)
        biases.append(b)
        print('conv2D')

    # ReLU after conv
    elif (i-1>=0) and isinstance(layer, nn.ReLU) and isinstance(layers[i-1], nn.Conv2d):
        print('ReLU')

    # sole ReLU
    elif isinstance(layer, nn.ReLU):
        print('sole ReLU - NOT IMPLEMENTED!!!')

    # flatten
    elif isinstance(layer, nn.Flatten):
        print('flatten')

    # max pool
    elif isinstance(layer, nn.MaxPool2d):
        mp_lip = utils.max_pool_lip(layer)
        print('max pool (lipshitz constant = ', mp_lip, ')', sep='')

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
