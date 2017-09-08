from spline import Spline
import numpy as np
import matplotlib.pyplot as plt


#d = np.array([[0,2],[1,3],[2,4],[3,3],[1,2],[0,0]])
#u = np.array([0, 1, 2, 3, 4, 5, 6, 7])
#d = np.array([[0,2],[1,3],[2,4],[3,3],[1,2],[0,0]])

# Line
d = np.array([[0, 1, 2, 3, 1, 0], [3, 3, 3, 3, 3, 3]])
u = np.array([0, 1, 2])

# Polygon 1
d = np.array([[0, 1, 2, 3, 1, 0], [2, 3, 4, 3, 2, 0]])
u = np.array([0, 1, 2, 3]) # 0 0 0 1 2 3 3 3


# Polygon 2 - ser lite skum ut
d = np.array([[0, 1, 2, 3, 1, 0, -1, -2], [2, 3, 4, 3, 2, 0, -1, -2]])
u = np.array([0, 1, 2, 3, 4, 5]) # 0 0 0 1 2 3 3 3



s = Spline(d,u)


s(d,u)

s.plotSpline()


x = 2
i = 3
N = s.basis(x,i)

x = np.linspace(0, 10)
plt.plot(x, np.sin(x), '--', linewidth=2)


# funkar inte, d är för kort
# Polygon 1
#d = np.array([[0, 1, 2, 3, 1, 0], [2, 3, 4, 3, 2, 0]])
#u = np.array([0, 1, 2, 3]) # 0 0 0 1 2 3 3 3
