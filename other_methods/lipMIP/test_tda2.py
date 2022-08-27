import sys 
sys.path.append('..')
import torch 
from pprint import pprint 

import utilities as utils 
from relu_nets import ReLUNet 
#import neural_nets.data_loaders as data_loaders
#import neural_nets.train as train 
from hyperbox import Hyperbox 
import interval_analysis as ia 
from lipMIP import LipMIP


DIMENSION = 2
simple_domain = Hyperbox.build_unit_hypercube(DIMENSION)
simple_c_vector = torch.Tensor([1.0, -1.0])
network_simple = ReLUNet([2, 16, 16, 2])
simple_prob = LipMIP(network_simple, simple_domain, simple_c_vector, verbose=True, num_threads=2)
inf_problem = LipMIP(network_simple, simple_domain, 'l1Ball1', verbose=True)
inf_problem.compute_max_lipschitz()
