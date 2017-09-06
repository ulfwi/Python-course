import numpy as np
import matplotlib.pyplot as plt

import math

k = 3  # degree of spline

class Spline:


    """
    u are node points
    d are deBoor points (in matrix form)
    """
    def __init__(self, d = None, N = 100):
        self.d = d
        # Create uniform knots, length of d+order of curve
        self.u = np.arange(len(self.d) - k + 1)
        self.u = np.append([self.u[0], self.u[0], self.u[0]], self.u)
        self.u = np.append(self.u, [self.u[-1], self.u[-1], self.u[-1]])
        # Create parameter vector t of length N
        self.t = np.arange(self.u[0], self.u[-1], self.u[-1]/N)

    @classmethod
    def interpolation(cls, x, y):
        #d =
        return cls(d)

    @classmethod
    def ctrlPoints(cls, d):
        return cls(d)

    def spline(self):
        spline_vec = np.array([])
        #deBoor
        return spline_vec


    def div(self,x,y):
        if y == 0:
            return 0
        return x / y

    # Returns basis function N_i^3 value in x
    def basis(self, x, i):
        # check if nodes coincide. use 0/0=0
        if k == 0:
            if self.u[i-1] <= x < self.u[i]:
                return 0.5 * (np.sign(x-self.u[i-1]) - np.sign(x-self.u[i])) # sum of two step functions
            else:
                return 0
        else:
            return self.div((x - self.u[i-1]),(self.u[i+k-1] - self.u[i-1])) * self.basis(x, i, k-1) \
                    + self.div((self.u[i+k] - x),(self.u[i+k] - self.u[i])) * self.basis(x, i+1, k-1)


    # Recursively evaluate the spline /deBoor algorithm
    def deBoor(self):
        return 0


    # polygon: boolean, plot if true
    def plotSpline(self, polygon):
        if polygon:
            plt.plot(self.d[:,0], self.d[:,1],'r-*')
        s = self.spline()
        plt.plot(s[:,0],s[:,1],'b')











