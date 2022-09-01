import numpy as np
from scipy.io import loadmat

def create_weights(net_dims, weight_type):
    # Create cell of weights for neural network with given dimensions
    #
    # params:
    #   * net_dims: list of ints    - dimensions of neural network
    #   * weights_type: str         - type of weights - in ['rand', 'ones']
    #
    # returns:
    #   W: list - weights of neural network

    num_layers = len(net_dims)-1

    W = [None]*num_layers
    for i in range(num_layers):
        if weight_type == 'ones':
            W[i] = ones(net_dims(i+1), net_dims(i))
            
        elif weight_type == 'rand':
            W[i] = (1 / np.sqrt(num_layers)) * np.random.randn(net_dims(i+1), net_dims(i))
            
        else:
            raise ValueError('[ERROR]: Please use weight_type in ["ones", "rand"]\n')
       
    return W


def load_weights(path):
    # Load weights from given path and extract network dimensions
    # 
    # params:
    #   * path: str - path of saved neural network weights
    #
    # returns:
    #   * weights: list          - loaded weights of neural network
    #   * net_dims: list of ints - dimensions of each layer in network
    # ---------------------------------------------------------------------

    # load weights from path
    dat = loadmat(path)
    weights = dat['weights'] # this is a numpy array of size (1,n_weights)
    weights = weights[0,:].tolist()
    
    # extract network dimensions from weights
    net_dims = [weights[0].shape[1]]
    for i in range(len(weights)):
        net_dims.append(weights[i].shape[0])

    return weights, net_dims


def split_weights(weights, net_dims, split_amount):
    # Splits neural network into subnetworks of size split_amount
    #
    # params:
    #   * weights: list             - weights of neural network
    #   * net_dims: list of ints    - dimensions of layers in network
    #   * split_amount: int         - size of each subnetwork
    #
    # returns:
    #   * split_w: list         - weights of each subnetwork
    #   * split_net_dims: list  - dimensions of each subnetwork
    # ---------------------------------------------------------------------

    # number of weights in neural network
    num_weights = len(weights)
    
    split_w = []
    split_net_dims = []
    
    counter = 1
    for k in range(0, num_weights, split_amount):
        
        # get ending index of split
        next_max_idx = k + split_amount - 1
        
        # if we exceed the total number of weights, cut this one short
        if next_max_idx > num_weights:
            next_max_idx = num_weights
        
        # add split section of weights to cell
        split_w.append(weights[k : next_max_idx + 1])
        split_net_dims.append(net_dims[k : next_max_idx + 2])
        counter = counter + 1

    return split_w, split_net_dims
