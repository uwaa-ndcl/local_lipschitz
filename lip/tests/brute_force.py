import numpy as np

def relu(x):
    return (x>0)*x

n = 3
n_samp = 10**8
X_ball = np.random.randn(n_samp, n)
norms = np.linalg.norm(X_ball, axis=1)
rr = np.random.rand(n_samp)**(1/n) # radii
X_ball = rr[:,None]*X_ball/norms[:,None]
X_ball = X_ball.T
X_ball_norms = np.linalg.norm(X_ball, axis=0)
print('max:', np.max(X_ball_norms), ' min:', np.min(X_ball_norms)) # sanity check

# values from Tiny Net
A_tiny = np.array([[-0.4249, -0.2224,  0.1548],
               [-0.0114,  0.4578, -0.0512],
               [ 0.1528, -0.1745, -0.1135]])
b_tiny = np.array([-0.5516, -0.3824, -0.2380])
x0_tiny = np.array([0.4963, 0.7682, 0.0885])

eps = 5.5
n_runs = 100
diffs = np.full(n_runs, np.nan)
for i in range(n_runs):
    # random system
    x0 = np.random.uniform(-1,1,n)
    #D = np.diag([0,1,1])
    D = np.eye(n)
    A = np.random.uniform(-1,1,(n,n))
    b = np.random.uniform(-1,1,n)
    y0 = A@x0 + b
    z0 = relu(y0)

    # analytical
    ybar = np.full(n, np.nan)
    r = np.full(n, np.nan)
    for j in range(n):
        ybar[j] = eps*np.linalg.norm(D @ A[j,:]) + y0[j]
        r[j] = (relu(ybar[j]) - relu(y0[j]))/(ybar[j] - y0[j])
    R = np.diag(r)
    lip_anl = np.linalg.norm(R@A@D, ord=2)

    #print(np.sum(r==0), '0s', np.sum(np.logical_and(r>0, r<1)), '0-1s', np.sum(r==1), '1s')

    # brute force
    DX = eps*X_ball    
    DX_norms = np.linalg.norm(DX, axis=0)
    X = x0[:,None] + DX
    Y = A@X + b[:,None]
    Z = relu(Y)
    fracs = np.linalg.norm(Z - z0[:,None], axis=0)/DX_norms
    lip_brute = np.max(fracs)

    diffs[i] = lip_anl - lip_brute

    print('analytical:', lip_anl, ' brute force:', lip_brute)
    if lip_brute>lip_anl:
        import pdb; pdb.set_trace()

import matplotlib.pyplot as pp
n, bins, patches = pp.hist(diffs, bins=10, facecolor='g', alpha=0.75)
pp.xlabel('L analytical - L brute (should always be positive!)')
pp.show()
