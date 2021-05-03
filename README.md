# Analytical Bounds on the Local Lipschitz Constants of ReLU Networks

---

This is the code for the paper

[**Analytical Bounds on the Local Lipschitz Constants of ReLU Networks**](https://arxiv.org/abs/2104.14672)

Trevor Avant & Kristi A. Morgansen


## dependencies

* Python 3 (we used version 3.9), Pytorch (we used version 1.8)

* the following non-standard python packages: torch, torchvision, numpy, PIL, scipy, tqdm, matplotlib


## using GPU or CPU

* This code can be run on either a CUDA-supported GPU, or on a CPU. This designation is set by uncommenting `device = 'cuda'` or `device = 'cpu'` in the `my_config.py` file.


## simulations

**lipschitz v. epsilon**

Run the following commands. To change the network, change `import mnist as exp` to something different. Note that the random and gradient methods may take a long time to run. You can toggle whether these methods are run by chagning the `compute_rand` and `compute_grad` variables.
* run `python calculate_bounds.py`
* run `python calculate_bounds_plot.py`


**adversarial bounds**: 

Run the following command. To change the network, change `import mnist as exp` to something different.
* `python calculate_bounds_adv.py`


## to run the 3rd party techniques

* lip estimation (Scaman et al., 2018): Run `lipestimation/custom.py` with the correct experiment name uncommented (e.g. `exp = 'alexnet')`. Then run `custom_get_sv.py` with the same network uncommented.

* LipSDP (Fazlyab et al., 2019): Run `lipsdp/run.py`. Note the code provided by the authors of LipSDP requires Matlab and the [Matlab Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).


## minimal working example / using your own network

A minimal working example showing how to compute the local Lipschitz bound of a feedforward network, and how to use that result to compute an adversarial bound, is shown in the `simple_example.py` file.


## extra

The `tests/` directory contains some additional files. These files include brute-force checks on some of the analytical results, as well as checks on some of our computational methods.
