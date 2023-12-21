
# if __name__ == '__main__':
#     # x = [x1..x100 ]
#     x = []
#     y = []

#     mu = 0
#     std = 10

#     Pb = []

#     Pc = scipy.special.softmax(np.random.normal(mu, std, 50))

#     np.random.seed(23)
#     Pt = scipy.special.softmax(np.random.normal(mu, std, 50))

#     l1 = []
#     l2 = []

#     alpha = 1
#     beta = 1
#     gamma = 1

#     nll = 0

#     for i in range(1, 101):
#         a_pb = scipy.special.softmax(np.random.normal(mu, std, size=50))
#         b_x = distance.jensenshannon(a_pb, Pc)
#         c_y = distance.jensenshannon(a_pb, Pt)

#         z_l1 = alpha * nll + beta * b_x - gamma * c_y
#         z_l2 = alpha * nll + beta * (max(0, b_x - c_y + (norm(Pc - Pt)))) ** 2


#         Pb.append(a_pb)
#         x.append(b_x)
#         y.append(c_y)
#         l1.append(z_l1)
#         l2.append(z_l2)

	 
#     fig = plt.figure()

#     ax = plt.axes(projection='3d')
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('P axis')
#     ax.set_zlabel('L2')

#     plt.plot(x, p, l2)
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import scipy
from scipy.spatial import distance
from numpy.linalg import norm

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
beta=gamma=1
l1=l2=x=y=p_b=[]

mu = 0
std = 10

np.random.seed(1)
Pc = scipy.special.softmax(np.random.normal(mu, std, 50))

np.random.seed(23)
Pt = scipy.special.softmax(np.random.normal(mu, std, 50))

for i in range(1, 1001):
    a_pb = scipy.special.softmax(np.random.normal(mu, std, size=50))
    b_x = distance.jensenshannon(a_pb, Pc)
    c_y = distance.jensenshannon(a_pb, Pt)

    z_l1 = beta * b_x - gamma * c_y
    z_l2 = beta * (max(0, b_x - c_y + (norm(Pc - Pt)))) ** 2

    p_b.append(a_pb)
    x.append(b_x)
    y.append(c_y)
    l1.append(z_l1)
    l2.append(z_l2)

# Plot the surface.

# Make data.
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)
R = np.sqrt(x**2 + y**2)
z = np.sin(R)

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
x,y = np.meshgrid(x,y)
# Customize the z axis.
ax.set_zlim(0,1.5) #(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()