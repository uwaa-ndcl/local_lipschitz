import os
import time
import numpy as np
import torch
from tqdm import tqdm

import my_config
import utils
import network_bound

# setup
device = my_config.device 
#plot_name = exp.plot_name
#layers = net.layers
#n_layers = len(layers)

def compute_bounds(exp, save_npz, compute_local=True, compute_global=True, compute_fgsm=True, compute_ifgsm=True, compute_gradasc=True, compute_cw=True):

    ###############################################################################
    # NETWORK AND SETUP
    # network and properties
    net = exp.net()
    net = net.to(device)
    net.eval()
    net_name = exp.net_name
    main_dir = exp.main_dir

    # iterate over each layer and get the output of each function
    x0 = exp.x0
    x0 = x0.to(device)
    layers = net.layers
    n_layers = len(layers)
    X = [x0]
    for i in range(n_layers):
        f = layers[i]
        X.append(f(X[-1]))

    # upper bound layer, global
    global_npz = os.path.join(main_dir, 'global.npz')
    dat = np.load(global_npz, allow_pickle=True)
    bound_layer_global = dat['bound']
    bound_global = np.prod(bound_layer_global)

    # get nominal output
    y0 = net(x0)
    top2_true, ind_top2_true = torch.topk(y0.flatten(), 2)
    ind1_true = ind_top2_true[0].item()
    ind2_true = ind_top2_true[1].item()

    # bounds, some may not be filled
    bound_local = []
    bound_global = []
    bound_fgsm = []
    bound_ifgsm = []
    bound_gradasc = []
    bound_cw = []

    ###############################################################################
    # LOCAL LIPSCHITZ LOWER BOUND
    # get largest and next largest elements of output (pre-softmax)
    if compute_local:
        print('\nLOWER ADVERSARIAL BOUND, LOCAL')

        t0 = time.time()
        x0.requires_grad = True 
        output_delta = (top2_true[0]-top2_true[1]).item() # minimum change in output to change classification

        # step 1: start with a guess,
        #         and determine if it is a lower or upper bound on the maximum epsilon
        pows = [-300, -100, -10, -7, -5, -3, -1, 0, 10, 1000]
        eps = [10**pow for pow in pows]
        i = 4 # starting guess on the index of eps
        L_bound = network_bound.local_bound(net, x0, eps[i], batch_size=exp.batch_size_l)
        if eps[i]*L_bound < output_delta/np.sqrt(2):
            eps_min = eps[i]
            i += 1
            search_dir = 'forward'
        else:
            eps_max = eps[i]
            i -= 1
            search_dir = 'backward'

        # step 2: determine the lower/upper bound if the guess was an upper/lower bound, respectively 
        if search_dir=='forward':
            while i<len(eps):
                L_bound = network_bound.local_bound(net, x0, eps[i], batch_size=exp.batch_size_l)
                if eps[i]*L_bound < output_delta/np.sqrt(2):
                    eps_min = eps[i]
                    i += 1
                else:
                    eps_max = eps[i]
                    break

            if 'eps_max' not in locals():
                print('ERROR: MAXIMUM EPSILON NOT FOUND!!!')

        elif search_dir=='backward':
            while i>=0:
                L_bound = network_bound.local_bound(net, x0, eps[i], batch_size=exp.batch_size_l)
                if eps[i]*L_bound < output_delta/np.sqrt(2):
                    eps_min = eps[i]
                    break
                else:
                    eps_max = eps[i]
                    i -= 1

            if 'eps_min' not in locals():
                print('ERROR: MINIMUM EPSILON NOT FOUND!!!')

        # step 3: refine the bounds using bisection
        '''
              |--------------------------|--------------------------|
           eps_min                      eps                      eps_max
        '''
        n_runs = 10
        for i in range(n_runs):
            eps = (eps_max + eps_min)/2
            L_bound = network_bound.local_bound(net, x0, eps, batch_size=exp.batch_size_l)
            if eps*L_bound < output_delta/np.sqrt(2):
                eps_greatest = eps
                #print('eps', eps, 'lower bound')
                #print('L bound', L_bound)
                eps_min = eps
            else:
                #print('eps', eps, 'NOT lower bound')
                #print('L bound', L_bound)
                eps_max = eps
        t1 = time.time()
        bound_local = eps_greatest
        print('bound:', bound_local)
        print('compute time:', t1-t0)

    ###############################################################################
    # GLOBAL LIPSCHITZ LOWER BOUND
    if compute_global:
        print('\nLOWER ADVERSARIAL BOUND, GLOBAL')
        # L >= eps_out/eps_in
        # eps_in >= eps_out/L
        #eps_max_class_global = output_delta/bound_global # old, wrong
        eps_max_class_global = (output_delta+np.sqrt(2))/bound_global
        bound_global = eps_max_class_global
        print('bound:', bound_global)

    ###############################################################################
    # FGSM
    if compute_fgsm:
        print('\nUPPER ADVERSARIAL BOUND, FGSM')
        
        # use bisection to find lowest working epsilon
        n_step = 10 # number of bisection steps
        if net_name=='mnist':
            eps_lo = 6e-1
            eps_hi = 7e-1
        elif net_name=='cifar10':
            eps_lo = 1e-1
            eps_hi = 2e-1
        elif net_name=='alexnet':
            eps_lo = 2e-2
            eps_hi = 3e-2
        elif net_name=='vgg16':
            eps_lo = 1e-1
            eps_hi = 2e-1
        
        i = 0
        eps = (eps_hi + eps_lo)/2
        bound_fgsm = np.nan
        while i<n_step:
            ind_fgsm, pert_norm = utils.fgsm(net, x0, eps)

            # epsilon works, make it smaller
            if ind_fgsm != ind1_true:
                bound_fgsm = pert_norm
                eps_hi = eps
                eps = (eps+eps_lo)/2

            # epsilon doesn't work, make it bigger
            else:
                eps_lo = eps
                eps = (eps+eps_hi)/2

            i += 1

        if np.isnan(eps):
            print('no fgsm perturbation found')
        else:
            print('bound:', pert_norm)

        '''
        eps_list = np.arange(0,100,.01)
        for eps in eps_list:
            ind_fgsm, pert_norm = utils.fgsm(net, x0, eps)
            if ind_fgsm != ind1_true:
                #print('minimum epsilon', eps)
                print('bound:', pert_norm)
                break

        if eps == eps_list[-1]:
            print('no perturbation found')
            pert_norm = np.nan

        bound_fgsm = pert_norm 
        '''


    ###############################################################################
    # I-FGSM
    if compute_ifgsm:
        print('\nUPPER ADVERSARIAL BOUND, IFGSM')

        if net_name=='mnist':
            eps = 1e-4
        elif net_name=='cifar10':
            eps = 1e-4
        elif net_name=='alexnet':
            eps = 3e-6
        elif net_name=='vgg16':
            eps = 5e-6

        ind_ifgsm, pert_norm, n_iters = utils.ifgsm(net, x0, eps, clip=False, lower=0, upper=0)
        if ind_ifgsm != ind1_true:
            bound_ifgsm = pert_norm
        else:
            print('no ifgsm perturbation found')
            bound_ifgsm = np.nan


    ###############################################################################
    # GRADIENT ASCENT
    if compute_gradasc:
        print('\nUPPER ADVERSARIAL BOUND, GRADIENT ASCENT')
        # evaluate the nominal input
        #x0.requires_grad = True
        #y0 = net(x0)
        #ind_true = top2_inds[0].item()
        #ind_true_2nd = top2_inds[1].item()
        #out_mask = torch.zeros(y0.shape).to(device)
        #out_mask[0,ind_true_2nd] = 1

        # gradient ascent
        if net_name=='mnist':
            step_size = .01 # best for no FGSM
            #step_size = .0001 # best for FGSM
        elif net_name=='cifar10':
            step_size = .001 # ind=2, step_size=.001
        elif net_name=='alexnet':
            step_size = .01 # ind=5, step_size=.01
        elif net_name=='vgg16':
            step_size = .01 # ind=3, step_size=.01

        # use only the sign of the gradient perturbation?
        fgsm = 0

        pert_min = np.inf
        for ind in tqdm(range(y0.numel())):
            x_i, pert_size_i, ind_new, its_i = utils.adv_asc_class_change(net, x0, ind, step_size, fgsm=fgsm)
            if (pert_size_i is not None) and (pert_size_i<pert_min):
                pert_min = pert_size_i
                ind_min = ind_new
                #print('new pert', pert_min)
                #print('its', its_i)

        bound_gradasc = pert_min
        print('bound:', bound_gradasc)

        # adversarial example info
        #print('grad:', grad)
        print('nominal input, class 1:', exp.classes[ind1_true])
        print('nominal input, class 2:', exp.classes[ind2_true])
        print('adversarial example, class 1:', ind_new)
        print('adversarial example, class 1 name:', exp.classes[ind_new])
        #print('iterations:', its)

    ###############################################################################
    # C&W
    if compute_cw:
        print('\nUPPER ADVERSARIAL BOUND, C&W')

        # gradient ascent
        if net_name=='mnist':
            lr = 5e-2
        elif net_name=='cifar10':
            lr = 1e-2 
        elif net_name=='alexnet':
            lr = 1e-1 
        elif net_name=='vgg16':
            lr = 1e-3

        # run the attack
        x_adv = utils.cw_attack(exp,LEARNING_RATE=lr,CONFIDENCE=0,PRINT_PROG=False)

        # analyze the results
        y0 = net(x0)
        class0 = torch.argmax(y0).item()
        y_adv = net(x_adv)
        class_adv = torch.argmax(y_adv).item()
        bound_cw = torch.norm(x0-x_adv).item()
        print('original class:', class0)
        print('new class:', class_adv)
        print('bound:', bound_cw) 

    np.savez(save_npz, bound_local=bound_local, bound_global=bound_global, bound_fgsm=bound_fgsm, bound_ifgsm=bound_ifgsm, bound_gradasc=bound_gradasc, bound_cw=bound_cw)
    """
    ###############################################################################
    # random
    eps = 5
    eps_inc = 5
    min_pert = None
    while min_pert is None:
        min_pert =  utils.adv_rand_class_change(net, x0, eps=eps, n_samp=exp.batch_size_ball)
        eps = eps + eps_inc
    print('\nSMALLEST RANDOM ADVERARIAL PERTURBATION FOUND', min_pert)
    """
