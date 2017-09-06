from spline import Spline
import numpy as np

d = np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]])
s = Spline(d)

x = 2
i = 3
N = s.basis(x,i)
print(N)


