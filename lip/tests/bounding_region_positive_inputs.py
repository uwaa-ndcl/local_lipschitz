# this script verifies the max output of matrix when the input is restricted to
# a unit ball or the non-negative part of a unit ball (see Appendix A.2)
import numpy as np

def sample_points(n, n_samp, mode):
    ''' sample from different norm balls '''
    
    if mode=='1':
        X = np.random.uniform(-1,1,(n,n_samp)) # this isn't uniform
        norms = np.linalg.norm(X, ord=1, axis=0)
        X = X/norms[None,:] 

    elif mode=='2':
        X = np.random.randn(n, n_samp)
        norms = np.linalg.norm(X, axis=0)
        rr = np.random.rand(n_samp)**(1/n) # radii
        X = rr[None,:]*X/norms[None,:]
        print('min of input vectors', np.min(X))

    elif mode=='inf':
        X = np.random.uniform(-1,1,(n,n_samp))
        
    X[:,-1] = 0 # make sure to include the origin
    return X


def get_max(ai, mode):
    ''' determine the x such that ||x|| <= 1 which maximizes ai^T x '''

    n = ai.size

    if mode=='1':
        ind_i_mx = np.argmax(np.abs(ai))
        x_opt = np.zeros(n)
        x_opt[ind_i_mx] = np.sign(ai[ind_i_mx])

    elif mode=='2':
        if np.linalg.norm(ai) != 0.0:
            x_opt = ai/np.linalg.norm(ai)
        else:
            x_opt = np.zeros(n)

    elif mode=='inf':
        x_opt = np.ones(n)
        x_opt[ai<0] = -1
        x_opt[ai==0] = 0

    return x_opt

# p-norm for unit ball
#mode = '1'
#mode = '2'
mode = 'inf'

# create a matrix A
m = 5
n = 3
A = np.random.uniform(-10,10,(m,n))
A_pls = A * (A>=0)
A_mns = -A * (A<=0)
A_err = np.linalg.norm(A-(A_pls-A_mns))
print('A breakdown error is', A_err)

# a bunch of positive random unit vectors
n_samp = 10**7
X = sample_points(n, n_samp, mode)

###############################################################################
# all inputs
print('ALL INPUTS')
# elementwise max output of matrix, sampling technique
Y = A @ X
ind_mx = np.argmax(Y, axis=1)
x_mx = X[:,ind_mx]
mx = np.max(Y, axis=1) 

# elementwise max output of matrix, my technique
print('sampling v. my technique error')
for i in range(m):
    ai = A[i,:]
    x_me = get_max(ai, mode)
    y_me = A @ x_me
    mx_me = y_me[i]
    print('element ' + str(i) + ':', np.linalg.norm(mx_me - mx[i]))

###############################################################################
# positive inputs only!
print('POSITIVE INPUTS')

# elementwise max output of matrix for positive inputs, sampling technique
X_pos = np.abs(X)
Y = A @ X_pos
ind_mx = np.argmax(Y, axis=1)
x_mx = X[:,ind_mx]
mx = np.max(Y, axis=1) 

# elementwise max output of matrix, my technique
print('sampling v. my technique error')
for i in range(m):
    ai_pls = A_pls[i,:]
    x_me = get_max(ai_pls, mode)
    y_me = A @ x_me
    mx_me = y_me[i]
    print('element ' + str(i) + ':', np.linalg.norm(mx_me - mx[i]))
