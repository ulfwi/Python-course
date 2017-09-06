import numpy as np

import math

class Spline:


    """
    u are node points
    d are deBoor points
    """
    def __init__(self, u, d):
        self.u = u
        self.d = d

    # """
    # x, y are function values
    # u are node points
    # """
    # def __init__(self, u, x, y):
    #     self.u = u
    #     self.x = x
    #     self.y = y

    def spline(self):
        spline_vec = np.array([])
        #deBoor
        return spline_vec


    # Creates basis functions N_i^k
    def basis(self, x, i, k=3):
        # cubic splines only defined on [u2,u_K-2]
        # check if nodes coincide. use 0/0=0
        if k == 0:
            if self.u[i-1] <= x < self.u[i]:
                return 0.5 * (np.sign(x-self.u[i-1]) - np.sign(x-self.u[i])) # sum of two step functions
            else:
                return 0
        else:
            return (x - self.u[i-1])/(self.u[i+k-1] - self.u[i-1]) * self.basis(x, i, k-1) \
                    + (self.u[i+k] - x)/(self.u[i+k] - self.u[i]) * self.basis(x, i+1, k-1)


    # Recursively evaluate the spline /deBoor algorithm
    def deBoor(self):
        return 0


    def plot(self):
        xplot = np.arange(self.x[0], self.x[-1], self.step)

