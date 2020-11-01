import numpy as np
import pypolycontain as pp
import matplotlib.pyplot as plt

x = np.array([4, 0]).reshape(2, 1)  # offset
G = np.array([[1, 0, 0.5], [0, 0.5, -1]]).reshape(2, 3)
C = pp.zonotope(x=x, G=G)
pp.visualize([C], title=r'$C$')
plt.show()

x = np.array([-1, -2]).reshape(2, 1)  # offset
G = np.array([[0, 1, 0.707, 0.293, 0.293, 0.707], [-1, 0, -0.293, 0.707, -0.707, 0.293]]).reshape(2, 6)
C = pp.zonotope(x=x, G=G)
pp.visualize([C], title=r'$C$')
plt.show()

H=np.array([[1,1],[-1,1],[0,-1], [2,3]])
h=np.array([1,1,0, 1])
A=pp.H_polytope(H,h)
# pp.visualize([A],title=r'$A$')

# D = pp.operations.intersection(A, A)
# pp.visualize([D])
# D=pp.operations.convex_hull(A,C)

# D = pp.operations.check_subset(C, C)
# D.color=(0.9, 0.9, 0.1)
# pp.visualize([D,A, C],title=r'$A$ (red),$C$ (blue), $D=A\oplus C$ (yellow)')
# t=np.array([5,0]).reshape(2,1) # offset
# theta=np.pi/6 # 30 degrees
# T=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]) # Linear transformation
# B=pp.AH_polytope(t,T,A)
# pp.visualize([B],title=r'$B$')

# pp.visualize([A,C,B],title=r'$A,B,C$')

# fig,ax=plt.subplots()
# fig.set_size_inches(6, 3)
# pp.visualize([A,C],ax=ax,fig=fig)
# ax.set_title(r'A triangle (red), rotated by 30 degrees (blue), and a zonotope (green)',FontSize=15)
# ax.set_xlabel(r'$x$',FontSize=15)
# ax.set_ylabel(r'$y$',FontSize=15)
# ax.axis('equal')
plt.show()