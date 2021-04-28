# Analytical Bounds on the Local Lipschitz Constants of ReLU Networks

---

This is the code for the paper

**Analytical Bounds on the Local Lipschitz Constants of ReLU Networks**

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

% You need to make a file like `tiny.py`, `mnist.py`, `cifar10.py`, `alexnet.py` or `vgg16.py`. It should have the following attributes: `x0` (the nominal input). It should also contain a function called `net()` which returns the network. The object returned by `net` should have a property called `layers` which is a list of layers in the network.


**adversarial bounds**: 

Run the following command. To change the network, change `import mnist as exp` to something different.
* `python calculate_bounds_adv.py`


## to run the 3rd party techniques

* to run lip estimation:
Run `custom.py` with the correct experiment name uncommented (e.g. `exp = 'alexnet')`. Then run `custom_get_sv.py` with the same network uncommented.

* to run lip SDP:


## minimal working example/using your own network

A minimal working example showing how to compute the local Lipschitz bound of a feedforward network is shown in the `simple_example.py` file.


## extra

## notes

* In this project,
