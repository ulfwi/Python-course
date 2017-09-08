from spline import Spline
import numpy as np
import matplotlib.pyplot as plt


#d = np.array([[0,2],[1,3],[2,4],[3,3],[1,2],[0,0]])
#u = np.array([0, 1, 2, 3, 4, 5, 6, 7])
d = np.array([[0,2],[1,3],[2,4],[3,3],[1,2],[0,0]])
d = np.array([[0, 1, 2, 3, 1, 0], [2, 3, 4, 3, 2, 0]])
u = np.array([0, 1, 2])
s = Spline(d,u)

print("beore")
s.plotSpline()
print("after")
x = 2
i = 3
N = s.basis(x,i)
print(N)

x = np.linspace(0, 10)
plt.plot(x, np.sin(x), '--', linewidth=2)


