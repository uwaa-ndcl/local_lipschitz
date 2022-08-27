import torch 

from relu_nets import ReLUNet 
from hyperbox import Hyperbox 
from lipMIP import LipMIP

import compnet as exp
net = exp.net()
net_seq = torch.nn.Sequential(*net.layers)
#n_input = 3
n_input = net_seq[0].in_features
#net_seq = torch.nn.Sequential(torch.nn.Linear(3,16,bias=True), torch.nn.ReLU(), torch.nn.Linear(16,16,bias=True), torch.nn.ReLU(), torch.nn.Linear(16,2,bias=True))
network = ReLUNet.from_sequential(net_seq)
domain = Hyperbox.build_unit_hypercube(n_input)

# old
#n_input = 2
#domain = Hyperbox.build_unit_hypercube(n_input)
#c_vector = torch.Tensor([1.0, -1.0])
#network_simp = ReLUNet([2, 16, 16, 2])

# compute
# 'l1ball1' means Lipschitz constant is computed w.r.t. infinity norm
# is this the only option that works?
linf_problem = LipMIP(network, domain, 'l1Ball1', verbose=True) 
linf_problem.compute_max_lipschitz()
#l1_problem = LipMIP(network, domain, c_vector, primal_norm='l1', verbose=True) 
#l1_problem.compute_max_lipschitz()
