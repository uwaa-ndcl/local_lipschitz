# this script is a sanity check to see if my method to randomly sample from the
# unit ball is correct
import numpy as np
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import lip.network.utils as ut

n_samp = 500
X2 = ut.sample_ball(2,n_samp)
X2 = X2.cpu().numpy()
X3 = ut.sample_ball(3,n_samp)
X3 = X3.cpu().numpy()
fig = pp.figure()
ax1 = fig.add_subplot(211)
ax1.scatter(X2[:,0], X2[:,1])
ax1.axis('equal')
print('2D maxes:', np.max(X2[:,0]), np.max(X2[:,1]))

ax2 = fig.add_subplot(212, projection='3d')
ax2.scatter(X3[:,0], X3[:,1], X3[:,2])
print('3D maxes:', np.max(X3[:,0]), np.max(X3[:,1]), np.max(X3[:,2]))

pp.show()
