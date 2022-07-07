# Analytical Bounds on the Local Lipschitz Constants of ReLU Networks

---

This is the code for the paper

[**Analytical Bounds on the Local Lipschitz Constants of ReLU Networks**](https://arxiv.org/abs/2104.14672)

Trevor Avant & Kristi A. Morgansen


## dependencies

* Python 3 (we used version 3.9), Pytorch (we used version 1.8)

* the following non-standard python packages: torch, torchvision, numpy, PIL, scipy, tqdm, matplotlib, torchattacks


## using GPU or CPU

This code can be run on either a CUDA-supported GPU, or on a CPU. This designation is set by uncommenting `device = 'cuda'` or `device = 'cpu'` in the `my_config.py` file.


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

* lip estimation (Scaman et al., 2018): Run `lipestimation/custom.py` with the correct experiment name uncommented (e.g. `exp = 'alexnet')`. Then run `custom_get_sv.py` with the same network uncommented.

* LipSDP (Fazlyab et al., 2019): Run `lipsdp/run.py`. Note the code provided by the authors of LipSDP requires Matlab and the [Matlab Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).


## extra

The `tests/` directory contains some additional scripts. These scripts include brute-force checks on some of the analytical results, as well as checks on some of our computational methods.
