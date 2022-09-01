import math
import itertools
import numpy as np
import cvxpy as cp
import scipy as sp
from scipy.linalg import block_diag
import mosek

import error_messages

def block_diag(arr_list):
    # create a block diagonal matrix from a list of cvxpy matrices

    # rows and cols of block diagonal matrix
    m = np.sum([arr.shape[0] for arr in arr_list])
    n = np.sum([arr.shape[1] for arr in arr_list])

    # loop to create the list for the bmat function
    block_list = []  # list for bmat function
    ind = np.array([0,0])
    for arr in arr_list:
        # index of the end of arr in the block diagonal matrix
        ind += arr.shape

        # list of one row of blocks
        horz_list = [arr]

        # block of zeros to the left of arr
        zblock_l = np.zeros((arr.shape[0], ind[1]-arr.shape[1]))
        if zblock_l.shape[1] > 0:
            horz_list.insert(0, zblock_l)

        # block of zeros to the right of arr
        zblock_r = np.zeros((arr.shape[0], n-ind[1]))
        if zblock_r.shape[1] > 0:
            horz_list.append(zblock_r)

        block_list.append(horz_list)

    B = cp.bmat(block_list)

    return B


def lipschitz_multi_layer(weights, mode, verbose, num_rand_neurons,
                          num_dec_vars, net_dims, network):
    # Computes Lipschitz constant of NN using LipSDP formulation
    # mode parameter is used to select which formulation of LipSDP to use
    #
    # params:
    #   * weights: cell          - weights of neural network in cell array
    #   * mode: str              - LipSDP formulation in ['network',
    #                             'neuron','layer','network-rand', 
    #                              'network-dec-vars']
    #   * verbose: logical       - if true, prints CVX output from solve
    #   * num_rand_neurons: int  - num of neurons to couple in 
    #                              LipSDP-Network-rand
    #   * num_dec_vars: int      - num of decision variables for
    #                              LipSDP-Network-Dec-Vars
    #   * net_dims: list of ints - dimensions of layers in neural net
    #   * network: struct        - data describing neural network
    #       - fields:
    #           (1) alpha: float            - slope-restricted lower bound
    #           (2) beta: float             - slope-restricted upper bound
    #           (3) weight_path: str        - path of saved weights of NN
    #                                         
    # returns:
    #   * L: float - Lipschitz constant of neural network
    # ---------------------------------------------------------------------

    L_sq = cp.Variable(nonneg=True)   # optval will be square of Lipschitz const

    # extract neural network parameters
    alpha = network['alpha']
    beta = network['beta']
    N = np.sum(net_dims[1:-1])     # total number of hidden neurons
    id = np.eye(N)

    # LipSDP-Network - one variable for each of the (N choose 2) neurons in
    # the network to parameterize T matrix.  This mode has complexity O(N^2)
    if mode == 'network':
        
        D = cp.Variable(shape=(N, 1), nonneg=True)
        zeta = cp.Variable(shape=(math.comb(N, 2), 1), nonneg=True)

        T = cp.diag(D)
        C = np.array(list(itertools.combinations(range(0,N), 2)))
        E = id[:, C[:, 0]] - id[:, C[:, 1]]
        T = T + E @ cp.diag(zeta) @ E.T

    # LipSDP-Network-Rand uses repeated nonlinearities with a random subset
    # of coupled neurons from the entire set of N choose 2 total neurons
    elif mode == 'network-rand':
        
        # cap number of random neurons
        num_rand_neurons = error_messages.cap_input(num_rand_neurons, N, 'randomly chosen neurons')

        D = cp.Variable(shape=(N, 1), nonneg=True)
        zeta = cp.Variable(shape=(num_rand_neurons, 1), nonneg=True)

        T = cp.diag(D)
        C = np.array(list(itertools.combinations(range(0,N), 2)))

        # take a random subset of neurons to couple
        k = np.random.permutation(C.shape[0])
        C = C[k[0:num_rand_neurons], :]

        # form T matrix using these randomly chosen neurons
        E = id[:, C[:, 0]] - id[:, C[:, 1]]
        T = T + E @ cp.diag(zeta) @ E.T
        
    # LipSDP-Network-Dec-Vars - uses repeated nonlinearities with a
    # spcified number of decision variables spaced out equally
    elif mode == 'network-dec-vars':
        
        # cap number of decision variables
        num_dec_vars = error_messages.cap_input(num_dec_vars, N, 'decision variables')
        
        D = cp.Variable(shape=(N, 1), nonneg=True)
        
        T = cp.diag(D)
        C = np.array(list(itertools.combinations(range(0,N), 2)))
        
        # space out decision variables in couplings
        spacing = int(np.ceil(math.comb(N, 2) / num_dec_vars))
        C = C[0::spacing, :]

        zeta = cp.Variable(shape=(C.shape[0], 1), nonneg=True)

        # form T matrix using these randomly chosen neurons
        E = id[:, C[:, 0]] - id[:, C[:, 1]]
        T = T + E @ cp.diag(zeta) @ E.T

    # LipSDP-Neuron - one CVX variable per hidden neuron in the network to 
    # parameterize T matrix.  This mode has complexity O(N).
    elif mode == 'neuron':

        D = cp.Variable(shape=(N, 1), nonneg=True)
        T = cp.diag(D)

    # LipSDP-Layer - one CVX variable per hidden hidden layer in the
    # network to parameterize T matrix.  This mode has complexity O(m)
    # where m is the number of hidden layers
    elif mode == 'layer':

        n_hid = len(net_dims) - 2
        identities = [None]*n_hid
        D = cp.Variable(shape=(n_hid,1), nonneg=True)

        for i in range(n_hid):
            identities[i] = D[i] * np.eye(net_dims[i+1])

        T = block_diag(identities)

    # If mode is not valid, raise error
    else:
        invalid_mode(mode)
       
    
    # Create Q matrix, which is parameterized by T, which in turn depends
    # on the chosen LipSDP formulation 
    Q = cp.bmat([[-2 * alpha * beta * T, (alpha + beta) * T], 
                 [(alpha + beta) * T, -2 * T]])

    # Create A term in Lipschitz formulation
    first_weights = sp.linalg.block_diag(*weights[0:-1])
    zeros_col = np.zeros((first_weights.shape[0], weights[-1].shape[1]))
    A = np.hstack((first_weights, zeros_col))

    # Create B term in Lipschitz formulation
    eyes = np.eye(A.shape[0])
    init_col = np.zeros((eyes.shape[0], net_dims[0]))
    B = np.hstack((init_col, eyes))

    # Stack A and B matrices
    A_on_B = np.vstack((A, B))

    # Create M matrix encoding Lipschitz constant
    weight_term = -1 * weights[-1].T @ weights[-1]
    middle_zeros = np.zeros((int(np.sum(net_dims[1:-2])), int(np.sum(net_dims[1:-2]))))
    lower_right = sp.linalg.block_diag(middle_zeros, weight_term)
    upper_left = L_sq * np.eye(net_dims[0])

    M = cp.bmat([[upper_left, np.zeros((upper_left.shape[0], lower_right.shape[1]))],
                 [np.zeros((lower_right.shape[0], upper_left.shape[1])), lower_right]])

    obj = cp.Minimize(L_sq)
    prob = cp.Problem(obj, [(A_on_B.T @ Q @ A_on_B) - M << 0])
    prob.solve(solver=cp.MOSEK, verbose=verbose)
    #prob.solve(solver=cp.CVXOPT, verbose=verbose)

    L = np.sqrt(prob.value)

    return L
