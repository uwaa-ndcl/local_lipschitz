'''
Proposition 4

this script is a brute-force verification that the equation for y_bar is correct
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
n_samp = 10**4 # number of vectors in unit ball
n_trials = 10**3 # number of times to regenerate the system
X_UNIT = sample_ball(n, n_samp).T
Y_BAR_DIFF = np.full((m,n_trials), np.nan)

# loop over trials
for i in range(n_trials):
    # system
    eps = np.random.uniform(.1,1)
    x0 = np.random.uniform(-1,1,n)
    b = np.random.uniform(-1,1,m)
    A = np.random.uniform(-1,1,(m,n))
    d = np.random.randint(0,2,n) 
    #d = np.ones(n)
    D = np.diag(d)
    y0 = A@x0 + b

    # analytical  
    aiTD_norm = np.linalg.norm(A@D, axis=1)
    y_bar_anl = eps*aiTD_norm + y0 

    # brute-force
    DELTA_X = eps*D@X_UNIT  
    X = x0[:,None] + DELTA_X
    Y = A@D@DELTA_X + y0[:,None]
    y_bar_brute = np.max(Y, axis=1)

    # compare
    Y_BAR_DIFF[:,i] = y_bar_anl - y_bar_brute

# print results
y_bar_diff_min = np.min(Y_BAR_DIFF)
y_bar_diff_max = np.max(Y_BAR_DIFF)
print('min and max y_bar diff, across all trials (should be non-negative!)')
print('min:', y_bar_diff_min)
print('max:', y_bar_diff_max)
