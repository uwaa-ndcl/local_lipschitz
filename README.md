# Analytical Bounds on the Local Lipschitz Constants of ReLU Networks

---

This is the code for the paper:

[**Analytical Bounds on the Local Lipschitz Constants of ReLU Networks**](https://ieeexplore.ieee.org/document/10164809)

Trevor Avant & Kristi A. Morgansen, *IEEE Transactions on Neural Networks and Learning Systems*, 2023


## dependencies

* Python 3 (we used version 3.9), Pytorch (we used version 1.8)

* the following non-standard python packages: torch, torchvision, numpy, PIL, scipy, tqdm, matplotlib


## using GPU or CPU

This code will automatically run on a CUDA-support GPU if you have one, and on a CPU otherwise. However, you can explicitly tell the code to run on the CPU by uncommenting `device = 'cpu'` in the `my_config.py` file.


## simulations

**lipschitz v. epsilon**

Run the following commands.
* `python calculate_bounds.py`
* `python calculate_bounds_plot.py`

Note that to change the network, change `import networks.mnist as exp` to something different. Also note that the random and gradient methods may take a long time to run. You can toggle whether these methods are run by chagning the `compute_rand` and `compute_grad` variables.


**adversarial bounds**: 

Run the following command:
* `python calculate_bounds_adv.py`

Note that to change the network, change `import networks.mnist as exp` to something different.


## minimal working example / using your own network

A minimal working example showing how to compute the local Lipschitz bound of a feedforward network, and how to use that result to compute an adversarial bound, is shown in the `simple_example.py` file.


## comparison with other methods

We compare our method to several other methods to compute/bound/estimate Lipschitz constants. The network we use is contained in the file `networks/compnet.py`. We were not able to test on LiPopt (Latorre et al., 2020) because it does not support networks with bias.

* Our Method: Run `$ python networks/compnet.py`.

* [SeqLip](https://github.com/avirmaux/lipEstimation) (Scaman & Virmaux, 2018): Run `other_methods/lipestimation/custom_get_sv.py` with the correct experiment name uncommented (e.g. `exp = 'alexnet')`. Then run `other_methods/lipestimation/custom.py` with the same network uncommented.

* [LipSDP](https://github.com/AntonXue/chordal-lipsdp) (Fazlyab et al., 2019): The original implementation of LipSDP uses Matlab. However, we will use a [Python implementation](https://github.com/trevoravant/LipSDP_python) which we wrote, which we have previously verified to produce the same results as the original Matlab version. First, run `$ python other_methods/make_weight_file.py` to generate the weight file. Then run `$ python other_methods/lipsdp_python/solve_sdp.py --form network --weight-path data/compnet/lipsdp/weights.mat` to produce the estimate.

* [lipMIP](https://github.com/revbucket/lipMIP) (Jordan & Dimakis, 2020): First, you need to install Gurobi (free for students) and the `gurobipy` Python package. Then run `python other_methods/lipMIP/run.py`.

* [RecurJac](https://github.com/huanzhang12/RecurJac-and-CROWN) (Zhang et al., 2019): This method requires Tensorflow. First, convert the Pytorch model to a Tensorflow model and save it by running `$ python other_methods/RecurJac-and-CROWN/convert_and_save_model.py`. Then, run  `$ python other_methods/RecurJac-and-CROWN/run.py` to calculate the Lipschitz constant.

## extra

The `tests/` directory contains some additional scripts. These scripts include brute-force checks on some of the analytical results, as well as checks on some of our computational methods.
