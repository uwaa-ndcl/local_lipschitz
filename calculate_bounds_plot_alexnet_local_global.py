import os
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.image as image
import torch
import torch.nn as nn

import my_config

#import tiny as exp
#import mnist as exp
#import cifar10 as exp
import alexnet as exp
#import vgg16 as exp

# plot settings
pp.rc('text', usetex=True)
pp.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts,amssymb}  \providecommand{\norm}[1]{\lVert#1\rVert}  \def\<#1>{\boldsymbol{\mathbf{#1}}}')
#scale = 10**9 # amount to scale y-axes to avoid the "x10^9" thing

# setup
device = my_config.device 
main_dir = exp.main_dir
net_name = exp.net_name
plot_name = exp.plot_name
net = exp.net()
net = net.to(device)
net.eval()
layers = net.layers
n_layers = len(layers)

# save files
rand_npz = os.path.join(main_dir, 'rand.npz')
grad_npz = os.path.join(main_dir, 'grad.npz')
global_npz = os.path.join(main_dir, 'global.npz')
local_npz = os.path.join(main_dir, 'local.npz')
est_lipest_npz = os.path.join(main_dir, 'lip_estimation/', 'results.npz')

#fig_i_png = os.path.join('fig/', net_name+'_layer_%1d.png') 
fig_png = os.path.join('fig/', net_name+'_local_global.png')
fig_pdf = os.path.join('fig/', net_name+'_local_global.pdf')

# lower bound, random
plot_rand = os.path.isfile(rand_npz)
if plot_rand:
    dat = np.load(rand_npz)
    eps_rand = dat['eps']
    bound_rand = dat['bound']

# upper bound, global
plot_global = 1
dat = np.load(global_npz, allow_pickle=True)
bound_layer_global = dat['bound']
bound_global = np.prod(bound_layer_global)

# lower bound, gradient
plot_grad = os.path.isfile(grad_npz)
if plot_grad:
    dat = np.load(grad_npz)
    eps_grad = dat['eps']
    bound_grad = dat['bound']

# estimate network, lip estimation
plot_net_lipest = os.path.isfile(est_lipest_npz)
if plot_net_lipest:
    dat = np.load(est_lipest_npz)
    autolip = dat['autolip'].item()
    greedy_seqlip = dat['greedy_seqlip'].item()
    if net_name=='alexnet':
        autolip *= 8 # account for max pooling Lipschitz constants
        greedy_seqlip *= 8 # account for max pooling Lipschitz constants

# upper bound, local
# always plot!
dat = np.load(local_npz, allow_pickle=True)
eps_local = dat['eps']
bound_local = dat['bound']

# get affine-relu indices
aff_relu_inds = []
for i, layer in enumerate(layers):
    if isinstance(layer, nn.Sequential):
        aff_relu_inds.append(i)

# get affine indices
aff_inds = []
for i, layer in enumerate(layers):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        aff_inds.append(i)

###############################################################################
# plot
lw = 4 # line width

fig, ax = pp.subplots(figsize=(6.4,4.8), dpi=300) # default figure size: 6.4, 4.8

# upper bound, global
if plot_global:
    ax.plot([exp.eps_min, exp.eps_max], [bound_global]*2, 'r', linewidth=lw,
            label='global')

# upper bound, local
ax.plot(eps_local, bound_local, color='b', linewidth=lw, label='local')
'''
# lower bound, gradient
if plot_grad:
    ax.plot(eps_grad, bound_grad, color='g', linestyle='--', linewidth=lw, dashes=(6,1), label='gradient (LB)')
'''
'''
# lower bound, random
if plot_rand:
    ax.plot(eps_rand, bound_rand, color='k', linestyle='--', linewidth=lw, dashes=(2,2), label='random (LB)')
'''
'''
if plot_net_lipest:
    ax.plot([exp.eps_min_net, exp.eps_max_net], [greedy_seqlip]*2, 'g--', label='estimate, Greedy SeqLip')
    ax.plot([exp.eps_min_net, exp.eps_max_net], [autolip]*2, 'b--', label='estimate, AutoLip')
'''

#pp.title(plot_name, fontsize=30)
ax.legend(loc=(.19,.1), fontsize=18)
#pp.yscale('log')
ax.tick_params(axis='both', labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(20)
pp.xlabel('input perturbation size ($\epsilon$)', fontsize=22)
pp.ylabel('Lipschitz bound', fontsize=22)
#pp.ylabel('Lipschitz bound ($\\times 10^9)$', fontsize=30)
ax.set_xlim(-.1, 3.1)
ax.set_ylim(-10**8, 1.25*10**9)

pp.tight_layout()

# image
ax_im = fig.add_axes([.59,.24,.4,.4]) # l, b, w, h
im = image.imread('data/imagenet/toucan.png')
ext = (0,1,0,1)
ax_im.imshow(im, aspect='equal', extent=ext)
ax_im.axis('off')
pp.text(.5, 1+.03, 'nominal input', ha='center', fontsize=14)
#pp.text(.5, -.08, 'network = AlexNet', ha='center', fontsize=14)

#pp.savefig(fig_png)
pp.savefig(fig_pdf)
#pp.show()
