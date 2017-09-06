import numpy as np

import math

class Spline:


    # (self, u, y, c, step = 0.01):
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


    # Recursively evaluate the spline /deBoor algorithm
    def deBoorPoints(self, u_index):
        u_short = self.u[2:-2]
        d = np.array([np.zeros(4)])
        alpha = np.array([np.zeros(4)])
        for i in range(0, np.len(d)-1):
            d[i] = self.c[u_index-3+i]
            alpha[i] = u[u_index] -
        #return d?

        # HUR VÄLJS CONTROL POINTS?? KOLLA MER PÅ DET.


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


    def plot(self):
        xplot = np.arange(self.x[0], self.x[-1], self.step)






