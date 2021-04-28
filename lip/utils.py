import warnings
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import my_config
device = my_config.device

def conv_matrix(conv, input_shape):
    '''
    get the matrix corresponding to a convolution function
    input_shape is of size (N,chan,rows,cols)
    '''

    # get input dimensions
    ch = input_shape[1]
    r = input_shape[2]
    c = input_shape[3]
    n = ch*r*c

    # copy conv and remove the bias
    conv_no_bias = copy.deepcopy(conv)
    conv_no_bias.bias = None
    
    # put identity matrix through conv function
    E = torch.eye(n).to(device)
    E = E.view(n,ch,r,c)
    A = conv_no_bias(E) # each row is a column
    A = A.view(n,-1).T

    return A


def conv_bias_vector(conv, output_shape):
    '''
    get the bias vector of a convolution function when expressed as an affine
    transformation
    '''

    # get bias
    bias = conv.bias.detach()
    b = torch.repeat_interleave(bias, output_shape[2]*output_shape[3])

    return b


def conv_trans_from_conv(conv):
    '''
    create a torch.nn.ConvTranspose2d() layer based on a torch.nn.Conv2d()
    layer (conv)
    '''

    conv_trans = nn.ConvTranspose2d(
            conv.out_channels,
            conv.in_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False)
    weight = conv.weight
    conv_trans.weight = torch.nn.Parameter(weight)

    return conv_trans


def get_RAD(func, input_shape, d=None, r_squared=None, n_iter=100):
    '''
    The largest singular value of matrix M can be found by taking the square
    root of largest eigenvlaue of the matrix P = M.T @ M. The largest
    eigenvalue of matrix M (which is the square of the largest singular value)
    can be found with a power iteration. The matrix P can also be found by
    applying a convolution operator to the image, and then applying a
    transposed convolution on that result.

    In this case we have:
        M = R @ A @ D
        M.T M = D.T @ A.T @ R.T @ R @ A @ D
              = D @ A.T @ R^2 @ A @ D

    Note that since we're using a power iteration, we are applying the
    operation:

    (D @ A.T @ R^2 @ A @ D) @ (D @ A.T @ R^2 @ A @ D) @ ...

    We can see that the two D's in the middle are redundant, so we only have to
    apply one of the D operations per iteration.

    func: function, either nn.Conv2d or nn.Linear
    input_shape for conv (shape of input array): =  batch, chan, H, W
    d: the diagonal elements of D, can also be None which means D=identity matrix
    r_squared: the diagonal elements of R^2, can be None which means R=identity matrix
    n_iter: number of iterations
    '''

    ########## conv2d ##########
    if isinstance(func, nn.Conv2d):

        # create conv trans layer
        conv = func
        conv_trans = conv_trans_from_conv(conv)

        # create new conv layer (which will have no bias)
        conv_no_bias = copy.deepcopy(conv)
        conv_no_bias.bias = None

        # determine batch size from zero_output_inds variable
        b, ch, n_row, n_col = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        # set batch size of d variable
        # do this here so we only have to negate d once
        if d is not None:
            d_not = torch.logical_not(d)
            if d_not.ndim == 1:
                d_not = d_not.repeat(b,1)

        # power iteration
        #torch.manual_seed(0)
        v = torch.rand(b*ch*n_row*n_col)
        v = v.to(device)
        if d is not None:
            v.view(b,-1)[d_not] = 0
        for i in range(n_iter):
            with torch.no_grad(): # this prevents out of memory errors
                # apply A
                V = torch.reshape(v, (b,ch,n_row,n_col)) # reshape to 4D array
                C1 = conv_no_bias(V) # output shape: (batch, out chan, H, W)

                # apply R^2
                if r_squared is not None:
                    C1_flat = C1.view(b,-1)
                    C1_flat *= r_squared

                # apply A.T
                C2 = conv_trans(C1, output_size=(b,ch,n_row,n_col))
                c2 = C2.view(b,-1) # reshape to 1D array

                # apply D
                if d is not None:
                    c2[d_not] = 0

                # normalize over each batch
                v = nn.functional.normalize(c2, dim=1)

        norm = torch.norm(c2, dim=1) # largest eigenvalue of M.T @ M
        spec_norm = torch.sqrt(norm) # largest singular value of M

    ########## fully-connnected ##########
    elif isinstance(func, nn.Linear):

        fc = func
        m,n = fc.weight.shape

        # create conv trans layer
        #conv_trans = conv_trans_from_conv(conv)
        fc_trans = copy.deepcopy(fc)
        fc_trans.weight = torch.nn.Parameter(fc_trans.weight.T)
        fc_trans.bias = None

        # create new conv layer (which will have no bias)
        fc_new = copy.deepcopy(fc)
        fc_new.bias = None

        # spectral norm of function
        b = 1

        # set batch size of d variable
        # do this here so we only have to negate d once
        if d is not None:
            d_not = torch.logical_not(d)
            if d_not.ndim == 1:
                d_not = d_not.repeat(b,1)

        # power iteration
        V = torch.rand(b,n)
        V = V.to(device)
        if d is not None:
            V[d_not] = 0
        for i in range(n_iter):
            with torch.no_grad(): # this prevents out of memory errors
                # apply A
                C1 = fc_new(V)

                # apply R^2
                if r_squared is not None:
                    C1 *= r_squared

                # apply A.T
                C2 = fc_trans(C1)

                # apply D
                if d is not None:
                    C2[d_not] = 0

                # normalize over each batch
                V = nn.functional.normalize(C2, dim=1)

        norm = torch.norm(C2, dim=1) # largest eigenvalue of M.T @ M
        spec_norm = torch.sqrt(norm) # largest singular value of M

    return spec_norm, V


def get_aiTD(func, input_shape, output_shape,
          pos_input=False, d=None, batch_size=2500):
    '''
    get the vector of values || a_i^T D || where a_i^T is the i^th row of A,
    for a convolution or fully-connected layer

    pos_input: boolean, whether or not the inputs to the function are positive
    d: 1D array, diagonal elements of D, None othewise
    '''

    # get sizes of A matrix
    n = input_shape.numel()
    m = output_shape.numel()
    '''
    # adaptive batch size
    numel_max = int(7e8)
    batch_size = np.max((numel_max//n, numel_max//m))
    batch_size = numel_max//m
    print('batch_size', batch_size)
    '''
    # convolution
    if isinstance(func, nn.Conv2d):
        # create conv trans layer
        conv = func
        conv_trans = conv_trans_from_conv(conv)

        # do this here so we only have to negate d once
        if d is not None:
            d_not = torch.logical_not(d)

        # create array E, where each row is a standard basis vector
        if batch_size > m:
            batch_size = m
        E_shape = (batch_size, output_shape[1], output_shape[2], output_shape[3])
        E = torch.eye(batch_size,m).to(device)
        E = E.view(E_shape).to(device)

        # loop over batches
        n_batch = int(np.ceil(m/batch_size))
        l = torch.empty(0).to(device)
        with torch.no_grad(): # this prevents out of memory errors
            for i in range(n_batch):
                ai = conv_trans(E, output_size=input_shape)
                ai = ai.view(batch_size, n)
                if pos_input:
                    ai[ai<0] = 0 # get positive part only
                if d is not None:
                    ai[:,d_not] = 0
                li = torch.norm(ai, dim=1)
                l = torch.cat((l, li))

                # shift the unit vectors for the next batch
                E_2d = E.view(batch_size, -1)
                E_2d = torch.roll(E_2d, batch_size, dims=1)
                E = E_2d.view(E_shape)

        l = l[:m] # chop off extra elements

    # fully-connected (these are usually small so we don't have to iterate)
    elif isinstance(func, nn.Linear):
        fc = func
        A = copy.deepcopy(fc.weight.data)

        # do this here so we only have to negate d once
        if d is not None:
            d_not = torch.logical_not(d)

        if pos_input:
            A[A<0] = 0 # get positive part only
        if d is not None:
            A[:,d_not] = 0
        l = torch.sqrt(torch.diag(A @ A.T))

    return l


def sample_ball(n, n_samp):
    '''
    sample points from inside an n-ball
    see (method 20): http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/ 
    also see: https://www.mathworks.com/matlabcentral/answers/439125-generate-random-sample-of-points-distributed-on-the-n-dimensional-sphere-of-radius-r-and-centred-at
    '''

    X_ball = torch.randn(n_samp, n).to(device)
    norms = torch.norm(X_ball, dim=1)
    rr = torch.rand(n_samp).to(device)**(1/n) # radii
    X_ball = rr[:,None]*X_ball/norms[:,None]

    return X_ball


def lower_bound_random(func, x0, eps, n_test=10000, batch_size=200):
    '''
    generate random points to find a naive lower bound of a conv-ReLU function
    func: nn.Conv2d or nn.Linear
    x0: nominal input, of size (1, shape_1, shape_2, ...)
    '''

    n = torch.numel(x0)

    # pass nominal point x0 through function
    y0 = func(x0)
    n_runs = int(np.ceil(n_test/batch_size))

    mx = -1 # max over all runs
    with torch.no_grad():
        for i in range(n_runs):
            # sample from inside unit ball
            X_ball = sample_ball(n, batch_size)
            X_ball *= eps
            X_ball_nrm = torch.norm(X_ball, dim=1)

            # pass ball points through aff-conv
            X_ball = X_ball.reshape((torch.Size([batch_size])+x0.shape[1:]))
            X = x0 + X_ball
            Y = func(X)
            diffs = Y - y0
            diffs = diffs.reshape((batch_size, -1))
            diffs_nrms = torch.norm(diffs, dim=1)
            nrms = diffs_nrms/X_ball_nrm # output diffs divided by input diffs
            nrms[torch.isinf(nrms)] = -float('inf') # turn infs (from zero X_ball_nrm values) to negative infs 
            nrms[torch.isnan(nrms)] = -float('inf') # turn nans into negative infs
            ind = torch.argmax(nrms)
            mx_i = nrms[ind]
            
            #if eps > 2: import pdb; pdb.set_trace()

            if mx_i > mx:
                mx = mx_i

    return mx


def lower_bound_random_many(func, x0, eps_lb, n_test=10000, batch_size=200):
    '''
    compute the lower bound of a convolution-relu function for many epsilons
    func: function
    x0: nominal input
    eps_lb: list of epsilons
    '''

    n_lb = eps_lb.size
    lb = np.full(n_lb, np.nan)
    for i in tqdm(range(n_lb)):
        lb[i] = lower_bound_random(func, x0, eps_lb[i], n_test=n_test, batch_size=batch_size)

    return lb


def get_upper_bounds(delta_alpha, sn):
    lub = sn[-1] # "looser upper bounds"
    tub = np.sum(delta_alpha*sn) # "tighter upper bounds"

    return lub, tub


def n_evenly_spaced_m(m, n):
    '''
    Bresenham's line algorithm
    take m evenly-spaced elements from a list of n elements
    outputs a list of indices
    see: https://stackoverflow.com/a/9873804/9357589
    '''

    return [i*n//m + n//(2*m) for i in range(m)]


def alpha_trans_sorted(b, eps, l, n_alpha):
    '''
    get a list of alpha values where Rbar changes (transitions)
    take only n_alpha evenly-spaced values from this list
    and return the corresponding delta alphas
    '''

    alpha_trans = -b/(eps*l)

    alpha_trans = torch.unique(alpha_trans) # remove duplicates
    alpha_trans, sort_inds = torch.sort(alpha_trans) # sort
    cond = torch.logical_and(alpha_trans>0, alpha_trans<1)
    alpha_trans = alpha_trans[cond]

    # add 1 to the alphas
    alpha_trans = torch.cat((alpha_trans, torch.tensor([1.0]).to(device)))
    n_alpha_trans = len(alpha_trans)
    inds = n_evenly_spaced_m(n_alpha, n_alpha_trans)
    inds = list(dict.fromkeys(inds)) # remove duplicates
    alpha_trans = alpha_trans[inds]
    alpha_trans[-1] = 1.0 # make sure 1 is the last alpha

    # get deltas
    alpha_trans_shft1 = torch.roll(alpha_trans, 1)
    alpha_trans_shft1[0] = 0
    alpha_trans_delta = alpha_trans - alpha_trans_shft1

    return alpha_trans, alpha_trans_delta


def jacobian_old(fun, x, row_inds=None, batch_size=10):
    '''
    calculate the Jacobian of a function with respect to input x
    the Jacobian will be an array in which the first dimension corresponds to
    the flattened output, and the remaining dimensions correspond to the input
    
    fun: pytorch function to compute the Jacobian with respect to
    x: input with batch size of 1
    row_inds: if is not None, take only a certain part of the Jacobian
              row_inds[0] is the index of the starting row & row_inds[1] is the index of the end row 
              consistent with Python indexing, row_inds[0] is inclusive and row_inds[1] is exclusive

    the code is partially based on:
    https://github.com/pytorch/pytorch/issues/10223#issuecomment-560564547
    '''

    # function input x
    x_shape = x.shape
    x_dim = len(x_shape)
    x.requires_grad_(True)

    # function output y
    y = fun(x)
    y_shape = y.shape
    n_y = torch.numel(y)

    # only getting certain rows of the Jacobian
    if row_inds is not None:
        ind_start = row_inds[0]
        ind_end = row_inds[1]

        if (ind_start < 0) or (ind_start > n_y):
            warnings.warn('start index out of range')
        if (ind_end < 0) or (ind_start > n_y):
            warnings.warn('end index out of range')

        n_jac_rows = ind_end - ind_start
        out_mask = torch.eye(n_jac_rows, n_y)
        out_mask = torch.roll(out_mask, ind_start, dims=1)
    else:
        n_jac = n_y
        out_mask = torch.eye(n_jac)

    # iterate through out_mask
    out_mask_batches = torch.split(out_mask, batch_size, dim=0)
    J = torch.Tensor().to(device) # empty tensor to start with
    for i, out_mask_i in enumerate(out_mask_batches):
        batch_size_i = out_mask_i.shape[0]
        new_shape = torch.tensor(y_shape).tolist()
        new_shape[0] = batch_size_i
        out_mask_i = out_mask_i.view(new_shape)

        # repeat list: list of how many times each dimension will be repeated
        # (all elements will be 1, except the 1st, which is the batch size)
        repeat_dims = [1]*x_dim
        repeat_dims[0] = batch_size_i
        x_rep = x.repeat(repeat_dims)

        # evaluate function and get Jacobian
        y = fun(x_rep)
        J_i = torch.autograd.grad(outputs=[y], inputs=[x_rep],
            grad_outputs=[out_mask_i], retain_graph=True)[0]
        J = torch.cat((J, J_i), dim=0)

    return J


def jacobian(fun, x, row_inds=None, batch_size=10):
    '''
    calculate the Jacobian of a function with respect to input x
    the Jacobian will be an array in which the first dimension corresponds to
    the flattened output, and the remaining dimensions correspond to the input
    
    fun: pytorch function to compute the Jacobian with respect to
    x: input with batch size of 1
    row_inds: if is not None, take only a certain part of the Jacobian
              row_inds[0] is the index of the starting row & row_inds[1] is the index of the end row 
              consistent with Python indexing, row_inds[0] is inclusive and row_inds[1] is exclusive

    the code is partially based on:
    https://github.com/pytorch/pytorch/issues/10223#issuecomment-560564547
    '''

    # function input x
    x_shape = x.shape
    x_dim = len(x_shape)
    x.requires_grad_(True)

    # function output y
    y = fun(x)
    y = torch.flatten(y, start_dim=1, end_dim=-1)
    n_y = torch.numel(y)

    # only getting certain rows of the Jacobian
    if row_inds is not None:
        ind_start = row_inds[0]
        ind_end = row_inds[1]
        if (ind_start < 0) or (ind_start > n_y):
            warnings.warn('start index out of range')
        if (ind_end < 0) or (ind_start > n_y):
            warnings.warn('end index out of range')
    else:
        ind_start = 0
        ind_end = n_y

    # get Jacobian one chunk at a time
    J = torch.Tensor().to(device) # empty tensor to start with
    i = ind_start
    while i<ind_end:
        i_end = np.min((i+batch_size, ind_end))
        batch_size_i = i_end-i

        # repeat list: list of how many times each dimension will be repeated
        # (all elements will be 1, except the 1st, which is the batch size)
        repeat_dims = [1]*x_dim
        repeat_dims[0] = batch_size_i
        x_rep = x.repeat(repeat_dims)

        # evaluate function and get Jacobian
        y = fun(x_rep)
        y = torch.flatten(y, start_dim=1, end_dim=-1)
        y_i = y[:,i:i_end]
        n_y_i = y_i.shape[1]
        mask_i = torch.eye(batch_size_i, n_y_i).to(device)
        J_i = torch.autograd.grad(outputs=[y_i], inputs=[x_rep],
            grad_outputs=[mask_i], retain_graph=True)[0]
        J = torch.cat((J, J_i), dim=0)

        i += batch_size

    return J


def jacobian_col(fun, x0, col_inds=None, batch_size=10):
    '''
    get only the certain columns of the jacobian referenced by col_inds
    col_inds refer to the jacobian reshaped into a matrix of size (output dim,input dim)

    fun: pytorch function to compute the Jacobian with respect to
    x: input with batch size of 1
    col_inds: col_inds[0] is the starting index of the column to return (inclusive)
              col_inds[1] is the ending index of the column to return (exclusive)
    
    '''

    J = torch.Tensor().to(device) # empty tensor to start with
    y0 = fun(x0)
    y0 = torch.flatten(y0, start_dim=1, end_dim=-1)
    n_y = y0.shape[1]

    i = 0
    while i<n_y:
        i_end = np.min((i+batch_size, n_y))
        Ji = jacobian(fun, x0, row_inds=[i,i_end], batch_size=batch_size)
        Ji = torch.flatten(Ji, start_dim=1, end_dim=-1)
        Ji = Ji[:,col_inds[0]:col_inds[1]]
        J = torch.cat((J, Ji), dim=0)
        i += batch_size

    return J


def jacobian_left_product(fun, x0, z, batch_size=10):
    '''
    get the product of a vector (z) and the jacobian (J) of fun at x0: z @ J

    fun: function to compute jacobian with respect to
    x0: input of original shape
    z: 1D vector
    '''
    
    vec = torch.Tensor().to(device) # empty tensor to start with
    n_x = x0.numel()

    # iterate over chunks of columns of jacobian
    i = 0
    while i<n_x:
        i_end = np.min((i+batch_size, n_x))
        J_cols = jacobian_col(fun, x0, col_inds=[i,i_end], batch_size=batch_size)
        vec_i = z @ J_cols
        vec = torch.cat((vec, vec_i), dim=0)
        i += batch_size

    return vec


def FGSM(fun, x, ind, eps, normalize=True):
    '''
    fast gradient sign method from "Explaining and Harnessing Adversarial Examples"
    https://arxiv.org/pdf/1412.6572.pdf
    
    ind = the index of the output to consider
    '''

    grad = jacobian(fun, x, row_inds=[ind, ind+1])
    if normalize:
        grad /= torch.norm(grad)
    x_pert = x + eps*torch.sign(grad)

    return x_pert


def FGM(fun, x, ind, eps, normalize=True):
    '''
    FGSM but without the sign

    ind = the index of the output to consider
    '''

    grad = jacobian(fun, x, row_inds=[ind, ind+1])
    if normalize:
        grad /= torch.norm(grad)
    x_pert = x + eps*grad

    return x_pert


def grad_adv(fun, x, eps, batch_size=10):
    '''
    adversarial perturbation with respect to the norm of all outputs
    (f^2)' = 2*f*f'
    '''

    y = fun(x)
    y = y.detach() # to avoid running out of memory
    y_vec = y.flatten()
    #J = jacobian(fun, x)
    #J_mat = J.view(J.shape[0], -1)
    #grad = 2 * y_vec @ J_mat
    grad = 2 * jacobian_left_product(fun, x, y_vec, batch_size=batch_size)
    grad = grad.view(x.shape)
    grad /= torch.norm(grad) # normalize
    x_pert = x + eps*grad

    return x_pert


def lower_bound_FGSM(fun, x, eps_lb, save_npz):
    '''
    compute lower bounds using FGSM
    '''

    n_eps = len(eps_lb)
    lb = np.full(n_eps, np.nan)
    y = fun(x)
    n_y = torch.numel(y)

    for i in range(n_eps):
        eps_i = eps_lb[i]
        lb_max = -1
        for j in range(n_y):
            x_pert = FGSM(fun, x, j, eps_i)
            y_pert = fun(x_pert) 
            lb_j = torch.norm(y - y_pert)/torch.norm(x - x_pert)
            if lb_j > lb_max:
                lb_max = lb_j
        lb[i] = lb_max
    np.savez(save_npz, eps=eps_lb, lb=lb)


def lower_bound_adv(fun, x, eps_lb, batch_size=10):
    '''
    compute lower bounds using the grad_adv() function
    '''

    n_eps = len(eps_lb)
    lb = np.full(n_eps, np.nan)
    y = fun(x)
    n_y = torch.numel(y)

    for i in range(n_eps):
        eps_i = eps_lb[i]
        x_pert = grad_adv(fun, x, eps_i, batch_size=batch_size)
        y_pert = fun(x_pert) 
        lb[i] = torch.norm(y - y_pert)/torch.norm(x - x_pert)

    return lb


def lower_bound_asc(fun, x, eps_lb, step_size=1e-4):
    '''
    compute lower bounds using gradient ascent
    '''

    # evaluate the nominal input
    y = fun(x)
    n_y = torch.numel(y)

    # gradient ascent
    n_step = 10**3
    xc = x
    eps_step = np.full(n_step, np.nan)
    lb_step = np.full(n_step, np.nan)
    for i in tqdm(range(n_step)):
        J = jacobian(fun, xc)
        J0 = J[0,:]
        pert = J0.view(x.shape)
        xc = xc + step_size*pert # "xc += pert" throws an error
        yc = fun(xc)
        eps_step[i] = torch.norm(x - xc)
        lb_step[i] = torch.norm(y - yc)/torch.norm(x - xc)

    # get largest lower bound for each epsilon
    n_eps = len(eps_lb)
    lb = np.full(n_eps, np.nan)
    for i in range(n_eps):
        eps_i = eps_lb[i]
        leq_mask = (eps_step<=eps_i.item())
        eps_leq = eps_step[leq_mask]
        lb_leq = lb_step[leq_mask]
        if lb_leq.size != 0:
            lb[i] = np.max(lb_leq)
        else:
            lb[i] = 0

    return lb 


def max_pool_inds(fun, input_shape, batch_size=100):
    '''
    brute force method to get the input indices of each max pool output

    x_list: each entry corresponds to an element of x, each entry is a
            list of indices of y that x appears in
    y_list: each entry corresponds to an element of y, each entry is a
            list of indices of x that contribute to the y
    '''

    x_test = torch.zeros(input_shape)
    y_test = fun(x_test)
    n_x = np.prod(input_shape)
    y_shape = y_test.shape
    n_y = y_test.numel()

    x_i = torch.eye(batch_size, n_x)
    x_i = x_i.view(batch_size, -1) 
    #x_batches = torch.split(x, batch_size, dim=0)

    x_list = []

    #https://stackoverflow.com/questions/12791501/python-initializing-a-list-of-lists
    y_list = [[] for i in range(n_y)]

    i = 0
    while i<n_x:
        if i+batch_size > n_x:
            len_i = n_x - i
            x_i = x_i[:len_i,:]

        y_i = fun(x_i.view(x_i.shape[0], *input_shape[1:]))
        y_i = y_i.view(y_i.shape[0], -1)

        for j, y_ij in enumerate(y_i):
            x_ij_ind = i+j

            # indices of where x shows up in y
            x_list_j = torch.where(y_ij==1)[0].tolist()
            #x_list_j = torch.where(y_ij!=0)[0].tolist()
            x_list.append(x_list_j)

            for y_active_ind in x_list_j:
                y_list[y_active_ind].append(x_ij_ind)

        i += batch_size
        x_i = torch.roll(x_i, batch_size, dims=1)

    return x_list, y_list


def max_pool_lip(fun):
    '''
    lipschitz constant of max poolng function the lipschitz constant is equal
    to the square root of the max number of output elments any input element
    can appear in

    How to determine the Lipschitz constant based on stride and kernel size:

    Consider a kernel positioned in 1D. The number of strides required to move
    the kernel to an entirely different set of inputs given by the equation

                        strides*stride_size >= kernel_size

    Solving for strides, we have

                        strides >= kernel_size/stride_size

    Since strides must be an integer, we can determine the minimum value of
    strides with the equation

                        strides = ceil(kernel_size/stride_size)

    This gives us the max number of times any imput can appear in the output.
    For a  2D max pooling function that is symmetric (equal kernel size and
    stride in each dimension), we can square n_max_1d to get the total n_max.
    '''
    
    if fun.dilation != 1:
        print('ERROR: NOT IMPLEMENTED FOR DILATIONS OTHER THAN 1')

    kernel_size = fun.kernel_size
    stride_size = fun.stride

    # get kernel size for each dimension
    if isinstance(kernel_size, int):
        kernel_size_0 = kernel_size
        kernel_size_1 = kernel_size
    elif isinstance(kernel_size, tuple):
        kernel_size_0 = kernel_size[0]
        kernel_size_1 = kernel_size[1]

    # get stride size for each dimension
    if isinstance(stride_size, int):
        stride_size_0 = stride_size
        stride_size_1 = stride_size
    elif isinstance(stride_size, tuple):
        stride_size_0 = stride_size[0]
        stride_size_1 = stride_size[1]

    n_max_0 = int(np.ceil(kernel_size_0/stride_size_0))
    n_max_1 = int(np.ceil(kernel_size_1/stride_size_1))
    n_max = n_max_0*n_max_1
    lip = np.sqrt(n_max)

    return lip


def adv_asc_class_change(net, x0, ind, step_size, fgsm=False, max_steps=10000):
    '''
    adversarial example via gradient ascent
    fgsm: boolean, use the sign of the gradient rather than the actual gradient
    '''

    y0 = net(x0)
    ind_true = torch.topk(y0.flatten(), 1)[1].item()

    out_mask = torch.zeros(y0.shape).to(device)
    out_mask[0,ind] = 1
    #out_mask[0,:] = 1

    x = x0
    i = 0
    while True:
        y = net(x)
        J = torch.autograd.grad(outputs=[y], inputs=[x],
            grad_outputs=[out_mask], retain_graph=True)[0]

        if fgsm:
            x = x + step_size*torch.sign(J)
        else:
            x = x + step_size*J # "xc += pert" throws an error
        y = net(x)
        
        top, ind_top = torch.topk(y.flatten(), 1)
        ind_top = ind_top.item()
        i+=1

        if (ind_top!=ind_true):
            pert_size = torch.norm(x-x0).item()
            return x, pert_size, ind_top, i

        if i>max_steps:
            return None, None, None, i


def adv_asc_class_change_batch(net, x0, step_size, fgsm=False, n_steps=1000, batch_size=25):
    '''
    adversarial example via gradient ascent
    try gradient ascent with respect to all indices
    fgsm: boolean, use the sign of the gradient rather than the actual gradient
    '''

    # initial
    x0 = x0
    y0 = net(x0)
    n_y = y0.numel()
    ind_true = torch.topk(y0.flatten(), 1)[1].item()
    X0 = torch.cat(batch_size*[x0])

    # batches
    batch_size = np.min([batch_size, n_y])
    n_batches = int(np.ceil(n_y/batch_size))
    out_mask = torch.eye(batch_size, n_y).to(device)

    pert_min = np.inf
    for j in tqdm(range(n_batches)):
        X = X0.clone()
        for i in range(n_steps):
            Y = net(X)
            J = torch.autograd.grad(outputs=[Y], inputs=[X],
                grad_outputs=[out_mask], retain_graph=True)[0]

            if fgsm:
                X = X + step_size*torch.sign(J)
            else:
                X = X + step_size*J # "xc += pert" throws an error
            Y = net(X)
            
            # which outputs produce a different classification?
            top, ind_top = torch.topk(Y, 1)
            ind_top = ind_top.flatten()
            delta_X = torch.flatten(X-X0, start_dim=1)
            perts = torch.norm(delta_X, dim=1)
            perts_dfrnt = perts[ind_top!=ind_true]

            # find lowest of classification
            if perts_dfrnt.nelement() > 0:
                pert_min_i = torch.min(perts_dfrnt).item()
                if pert_min_i < pert_min:
                    pert_min = pert_min_i

        out_mask = torch.roll(out_mask, batch_size)

    return pert_min


def adv_rand_class_change(net, x0, eps, n_samp=10000):
    '''
    randomly sample the unit ball, and find the minimum perturbation that
    changes the class
    '''

    n = x0.numel()
    y0 = net(x0)
    ind_x0 = torch.topk(y0.flatten(), 1)[1].item()

    DELTA_X = eps*sample_ball(n,n_samp)
    DELTA_X = torch.reshape(DELTA_X, torch.Size([n_samp])+x0.shape[1:])
    DELTA_X_NRM = torch.norm(DELTA_X.view(n_samp,-1), dim=1)
    X = x0 + DELTA_X
    Y = net(X)
    
    ind_X = torch.topk(Y, 1)[1].flatten()

    NRM_CLASS_CHANGE = DELTA_X_NRM[ind_x0!=ind_X]

    if len(NRM_CLASS_CHANGE)==0:
        return None

    else:
        min_pert = torch.min(NRM_CLASS_CHANGE).item()
        return min_pert


def fgsm_new(net, x0, eps):
    '''
    https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    '''

    # nominal input and output
    x0.requires_grad = True
    y0 = net(x0)
    ind_true = torch.topk(y0.flatten(), 1)[1].item()

    # loss criterion
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = nn.BCELoss()
    #target = torch.zeros(1,10).to(device)
    #target[0,ind_true] = 1

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    target = torch.tensor([ind_true], dtype=torch.long).to(device)

    loss = criterion(y0, target)
    net.zero_grad()
    loss.backward()

    # get gradient and create perturbation
    grad = x0.grad.data
    pert = eps*torch.sign(grad)
    pert_norm = torch.norm(pert).item()
    x_new = x0 + pert
    y_new = net(x_new)
    
    ind_new = torch.topk(y_new.flatten(), 1)[1].item()
    #print('true ind', ind_true)
    #print('pert ind', ind_new)
    #print(y0)
    #print(y_new)

    return ind_new, pert_norm
