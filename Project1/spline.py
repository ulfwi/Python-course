import numpy as np

import math

class Spline:

    # u are node points
    # y are node function values
    # d are deBoor points
    # x
    # N
    def __init__(self, u, y, step = 0.01):
        self.u = u
        self.y = y
        self.step = step
        length = u[-1] - u[0]

    def spline(self):
        spline_vec = np.array([])
        #deBoor
        return spline_vec


    # Creates basis functions N_i^k
    def basis(self, x, i):
        N_0 = np.array([])
        Heaviside = 0.5 * (np.sign(x-self.u[i-1]) - np.sign(x-self.u[i]))


    # Recursively evaluate the spline /deBoor algorithm
    def deBoor(self):
        #return d?

    def plot(self):
        xplot = np.arange(self.x[0], self.x[-1], self.step)

        plot()