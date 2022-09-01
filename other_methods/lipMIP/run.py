import torch 

from relu_nets import ReLUNet 
from hyperbox import Hyperbox 
from lipMIP import LipMIP

import networks.tiny as exp
#import networks.compnet as exp

net = exp.net()
net = net.eval()
net_seq = torch.nn.Sequential(*net.layers)
network = ReLUNet.from_sequential(net_seq)

n_input = exp.n_input
domain = Hyperbox.build_unit_hypercube(n_input)

# compute
# 'l1ball1' means Lipschitz constant is computed w.r.t. infinity norm
# is this the only option that works?
linf_problem = LipMIP(network, domain, 'l1Ball1', verbose=True) 
linf_problem.compute_max_lipschitz()
