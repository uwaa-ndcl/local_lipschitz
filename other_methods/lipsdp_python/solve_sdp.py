import argparse
import numpy as np
from scipy.io import savemat
import os
from time import time

import solve_LipSDP

def main(args):

    start_time = time()

    network = {
        'alpha': args.alpha,
        'beta': args.beta,
        'weight_path': args.weight_path,
    }

    lip_params = {
        'formulation': args.form,
        'split': args.split,
        'parallel': args.parallel,
        'verbose': args.verbose,
        'split_size': args.split_size,
        'num_neurons': args.num_neurons,
        'num_workers': args.num_workers,
        'num_dec_vars': args.num_decision_vars
    }

    L = solve_LipSDP.solve_LipSDP(network, lip_params)
    print(f'LipSDP-{args.form.capitalize()} gives a Lipschitz constant of {L:.3f}')
    print(f'Total time: {float(time() - start_time):.5} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--form',
        default='neuron',
        const='neuron',
        nargs='?',
        choices=('neuron', 'network', 'layer', 'network-rand', 'network-dec-vars'),
        help='LipSDP formulation to use')

    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='prints CVX output from solve if supplied')

    parser.add_argument('--alpha',
        type=float,
        default=0,
        help='lower bound for slope restriction bound')

    parser.add_argument('--beta',
        type=float,
        default=1,
        help='lower bound for slope restriction bound')

    parser.add_argument('--num-neurons',
        type=int,
        default=100,
        help='number of neurons to couple for LipSDP-Network-rand formulation')

    parser.add_argument('--split',
        action='store_true',
        help='splits network into subnetworks for more efficient solving if supplied')

    parser.add_argument('--parallel',
        action='store_true',
        help='parallelizes solving for split formulations if supplied')

    parser.add_argument('--split-size',
        type=int,
        default=2,
        help='number of layers in each subnetwork for splitting formulations')

    parser.add_argument('--num-workers',
        type=int,
        default=0,
        help='number of workers for parallelization of splitting formulations')

    parser.add_argument('--num-decision-vars',
        type=int,
        default=10,
        help='specify number of decision variables to be used for LipSDP')

    parser.add_argument('--weight-path',
        type=str,
        required=True,
        help='path of weights corresponding to trained neural network model')

    args = parser.parse_args()

    if args.parallel is True and args.num_workers < 1:
        raise ValueError('When you use --parallel, --num-workers must be an integer >= 1.')

    if args.split is True and args.split_size < 1:
        raise ValueError('When you use --split, --split-size must be an integer >= 1.')

    main(args)
