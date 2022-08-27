# Analytical Bounds on the Local Lipschitz Constants of ReLU Networks

---

This is the code for the paper

[**Analytical Bounds on the Local Lipschitz Constants of ReLU Networks**](https://arxiv.org/abs/2104.14672)

Trevor Avant & Kristi A. Morgansen


## dependencies

* Python 3 (we used version 3.9), Pytorch (we used version 1.8)

* the following non-standard python packages: torch, torchvision, numpy, PIL, scipy, tqdm, matplotlib


## using GPU or CPU

This code can be run on either a CPU or CUDA-supported GPU. If you have a CUDA-supported GPU, it will automatically be used. If you want to manually select which device to use, you can uncomment `device = 'cuda'` or `device = 'cpu'` in the `my_config.py` file.


## simulations

**lipschitz v. epsilon**

Run the following commands.
* `python calculate_bounds.py`
* `python calculate_bounds_plot.py`

Note that to change the network, change `import mnist as exp` to something different. Also note that the random and gradient methods may take a long time to run. You can toggle whether these methods are run by chagning the `compute_rand` and `compute_grad` variables.


**adversarial bounds**: 

Run the following command:
* `python calculate_bounds_adv.py`

Note that to change the network, change `import mnist as exp` to something different.


## minimal working example / using your own network

A minimal working example showing how to compute the local Lipschitz bound of a feedforward network, and how to use that result to compute an adversarial bound, is shown in the `simple_example.py` file.


## comparison with other methods

* lip estimation (Scaman et al., 2018): Run `other_methods/lipestimation/custom.py` with the correct experiment name uncommented (e.g. `exp = 'alexnet')`. Then run `custom_get_sv.py` with the same network uncommented.

* LipSDP (Fazlyab et al., 2019): Run `lipsdp/run.py`. Note the code provided by the authors of LipSDP requires Matlab and the [Matlab Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

* LipSDP (Chordal): You must have Julia and Mosek (free for students) installed. Run `$ julia other_methods/chordal-lipsdp/scripts/install_pkgs.jl` to install required packages. Then run `$ julia -i other_methods/chordal-lipsdp/scripts/run_nnet.jl --nnet !!!`. Then in the Julia prompt run `soln, lipconst = solveLipschitz(ffnet, weight_scales, :lipsdp)`. Once that command completes, the solution can then be found by entering `soln` in the Julia prompt. 
see: https://github.com/AntonXue/chordal-lipsdp

* lipMIP: First, you need to install Gurobi (free for students) and the `gurobipy` Python package.

## extra

The `tests/` directory contains some additional scripts. These scripts include brute-force checks on some of the analytical results, as well as checks on some of our computational methods.
