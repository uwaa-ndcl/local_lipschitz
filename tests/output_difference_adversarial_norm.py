'''
Proposition 6

this script is a brute-force verification that the equation we derived which
relates local Lipschitz constants, input perturbations, and adversarial bounds
is true:
               epsilon*L(x_0,X) < delta/sqrt(2)
'''
import numpy as np

# setup
n_dim = 3
n = 10**7
y0 = np.random.rand(n_dim)

# the largest elements of y0
y0_sort = np.sort(y0)
y0_top1 = y0_sort[-1]
y0_top2 = y0_sort[-2]
ind_y0_max = np.argmax(y0)

# make a bunch of y vectors
Y = np.random.rand(n_dim, n)

# for which y vectors does the max (i.e. classification) change?
Y_min_y0 = Y - y0[:,None]
ind_Y_max = np.argmax(Y, axis=0)
inds_new_class = np.not_equal(ind_y0_max, ind_Y_max)

# compute difference norms
norms = np.linalg.norm(Y_min_y0, axis=0)

# take only the y vectors values which have a different max than y0
norms = norms[inds_new_class]
Y = Y[:,inds_new_class]

# get the indices of the y's with the lowest norms
ind0, ind1 = np.argpartition(norms, 2)[:2]
diff_lowest = norms[ind0]
diff_lowest_analytical = (y0_top1 - y0_top2)/np.sqrt(2)

# print
print('y0', y0)
print('y of lowest ||y - y_0||', Y[:,ind0])
print('lowest ||y - y_0||', diff_lowest)
print('lowest ||y - y_0|| analytical', diff_lowest_analytical)
print('lowest ||y - y_0|| error (should be small and positive)', diff_lowest - diff_lowest_analytical)
