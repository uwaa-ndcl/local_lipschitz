'''
Theorem 1

this script is a brute-force verification that the scalar local Lipschitz
result is true
'''
import numpy as np

def relu(y):
    return (y>0)*y

def lip(y0,y):
    return np.abs(relu(y) - relu(y0))/np.abs(y-y0)

# setup
n_trials = 10**4
n_samps = 7
lip_anl = np.full(n_trials, np.nan)
lip_brute = np.full(n_trials, np.nan)

# loop over many trials
for i in range(n_trials):
    # sample some points
    y0 = np.random.uniform(-1,1,1)
    Y = np.random.uniform(-1,1,n_samps)

    # analytical
    ybar = np.max(Y)
    lip_anl[i] = lip(y0,ybar)

    # brute-force
    lip_frac = lip(y0,Y)
    lip_brute[i] = np.max(lip_frac)

# compare analytical and brute-force results
lip_diff = lip_anl - lip_brute
lip_err = np.linalg.norm(lip_diff)

# print results
#print(lip_anl)
#print(lip_brute)
print('analytical v brute-force local Lipschitz error (should be zero):')
print(lip_err)
