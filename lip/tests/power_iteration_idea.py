import numpy as np

# matrix
n = 2
A = np.array([[1,-3],[4,-7]])
#A = np.random.rand(n,n)
eig_mx_nrm = np.linalg.norm(A, ord=2)
print('eig max norm', eig_mx_nrm)

# sample some points in the unit sphere
n_pts = 10**4
x = np.random.randn(n, n_pts)
nrm = np.linalg.norm(x,axis=0)
x = x/nrm[None,:]
Ax = A@x
Ax_nrm = np.linalg.norm(Ax, axis=0)
eig_mx_pts = np.max(Ax_nrm)
print('eig max sampling', eig_mx_pts)
'''
# sanity check plot
import matplotlib.pyplot as pp
pp.figure()
pp.scatter(x[0,:], x[1,:])
pp.show()
'''
# power iteration
n_iter = 1000
v = np.random.rand(n,1)
for i in range(n_iter):
    v = A @ v
    v = v/np.linalg.norm(v)
eig_mx_pi = np.linalg.norm(A@v)
print('eig max power it', eig_mx_pi)

# positive sampling
x_pos = np.abs(x)
Ax_pos = A@x_pos
eig_mx_pts_pos = np.max(Ax_pos)
print('eig max pos sample', eig_mx_pts_pos) 

# positive power iteration
n_iter = 10
v = np.random.rand(n,1)
for i in range(n_iter):
    v = A @ v
    v = v/np.linalg.norm(v)
    print(v)
    v[v<0] = 0
eig_mx_pi_pos = np.linalg.norm(A@v)
print('eig max power it pos', eig_mx_pi_pos)
