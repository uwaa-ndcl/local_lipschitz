import numpy as np
import multiprocessing
from itertools import repeat

import lipschitz_multi_layer

def compute_L_split(k, split_W, split_net_dims, mode, verbose, num_rand_neurons, num_dec_vars, network):
    # compute Lipschitz constant of a subnetwork after splitting

    curr_weights = split_W[k]
    curr_net_dims = split_net_dims[k]

    if len(curr_weights) > 1:
        Lf_reduced_piece = lipschitz_multi_layer.lipschitz_multi_layer(
            curr_weights, mode, verbose, num_rand_neurons, num_dec_vars,
            curr_net_dims, network)

    # if there is only one matrix in this layer, just
    # multiply by the norm of that matrix
    else:
        Lf_reduced_piece = np.linalg.norm(curr_weights[0], ord=2)

    return Lf_reduced_piece


def split_and_solve(split_W, split_net_dims, lip_params, network):
    # Compute Lipschitz constant as product of constants from subnetworks
    # Can be run in parallel by specifying parallel and num_workers flags
    # in lip_params struct
    #
    # params:
    #   * split_W: list         - weights for each subnetwork
    #   * split_net_dims: list  - dimensions of layers in each subnetwork
    #   * network: dictionary       - data describing neural network
    #       - fields:
    #           (1) alpha: float            - slope-restricted lower bound
    #           (3) beta: float             - slope-restricted upper bound
    #           (3) weight_path: str        - path of saved weights of NN
    #   * lip_params: dictionary    - parameters for LipSDP
    #       - fields:
    #           (1) formulation: str    - LipSDP formulation to use
    #           (2) split: logical      - if true, use splitting 
    #           (3) parallel: logical   - if true, parallelize splitting
    #           (4) verbose: logical    - if true, print CVX output
    #           (5) split_size: int     - size of subnetwork for splitting
    #           (6) num_neurons: int    - number of neurons to couple in
    #                                     LipSDP-Network-Rand mode
    #           (7) num_workers: int    - number of workers for parallel-
    #                                     ization of splitting formulations
    #           (8) num_dec_vars: int   - number of decision variables for
    #                                     LipSDP-Network-Dec-Vars
    #
    # returns:
    #   * lip_prod: float - Lipschitz constant found by splitting network
    #                       into subnetworks and solving LipSDP for each
    #                       piece
    # ---------------------------------------------------------------------

    # number of subnetworks after splitting
    num_splits = len(split_W)
    
    # unpack variables from lip_params
    mode = lip_params['formulation']
    verbose = lip_params['verbose']
    num_rand_neurons = lip_params['num_neurons']
    num_dec_vars = lip_params['num_dec_vars']

    # parallel
    # for loop is parallelizable 
    if lip_params['parallel']:
        pool = multiprocessing.Pool(processes=lip_params['num_workers'])
        Lf_reduced_pieces = pool.starmap(
            compute_L_split, zip(range(num_splits), repeat(split_W),
            repeat(split_net_dims), repeat(mode), repeat(verbose),
            repeat(num_rand_neurons), repeat(num_dec_vars), repeat(network)))
        pool.close()
        pool.join()
        lip_prod = np.prod(Lf_reduced_pieces)

    # not parallel 
    else:
        # initialize Lipschitz constant of network
        lip_prod = 1
        for k in range(num_splits):

            Lf_reduced_piece = compute_L_split(
                k, split_W, split_net_dims, mode, verbose, num_rand_neurons,
                num_dec_vars, network)

            # update product
            lip_prod = lip_prod * Lf_reduced_piece

    return lip_prod
