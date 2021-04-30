'''
Theorem 2

this script is a brute-force verification that the affine-ReLU local Lipschitz
bound is true
'''
import numpy as np

def relu(y):
    return (y>0)*y

def sample_ball(n, n_samp):
    '''
    sample points from inside an n-ball
    '''

    X_ball = np.random.randn(n_samp, n)
    norms = np.linalg.norm(X_ball, axis=1)
    rr = np.random.rand(n_samp)**(1/n) # radii
    X_ball = rr[:,None]*X_ball/norms[:,None]

    return X_ball

# setup
n = 3
m = 4
n_samp = 10**4
n_trials = 10**3
X_UNIT = sample_ball(n, n_samp).T

# loop over trials
lip_anl = np.full(n_trials, np.nan)
max_frac = np.full(n_trials, np.nan)
for i in range(n_trials):
    # create random system
    eps = np.random.uniform(.1,1)
    x0 = np.random.uniform(-1,1,n)
    A = np.random.uniform(-1,1,(m,n))
    b = np.random.uniform(-1,1,m)
    d = np.random.randint(0,2,n) 
    #d = np.ones(n)
    D = np.diag(d)

    # analytical solution
    y0 = A @ x0 + b
    z0 = relu(y0)
    aiTD_norm = np.linalg.norm(A @ D, axis=1)
    ybar = eps*aiTD_norm + y0
    np.seterr(invalid='ignore')
    r = (relu(ybar) - relu(y0))/(ybar - y0)
    zero_inds = (aiTD_norm==0) 
    r[zero_inds] = 0
    R = np.diag(r) 
    lip_anl[i] = np.linalg.norm(R @ A @ D)

    # brute-force solution
    DELTA_X = eps*D@X_UNIT  
    X = x0[:,None] + DELTA_X
    Z = relu(A @ X + b[:,None])
    lip_num = np.linalg.norm(Z - z0[:,None], axis=0)
    lip_den = np.linalg.norm(X - x0[:,None], axis=0)
    lip_frac = lip_num/lip_den
    max_frac[i] = np.max(lip_frac)
    if np.all(lip_den==lip_num):
        max_frac[i] = 0

# compare
lip_diff = lip_anl - max_frac
print('these should all be positive!')
#print(lip_diff)
print('min', np.min(lip_diff))
print('max', np.max(lip_diff))
