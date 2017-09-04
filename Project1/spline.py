import numpy as np

import math

class Spline:

    # x are node points (u)
    # y are node function values
    # d are deBoor points
    # N
    def __init__(self, x, y, step = 0.01):
        self.x = x
        self.y = y
        self.step = step

    def spline(self):
        spline_vec = np.array([]);
        #deBoor
        return spline_vec


    # Creates basis functions N_i^k
    def basis(self, i, k):

    # Recursively evaluate the spline /deBoor algorithm
    def deBoor(self):
        #return d?

    def plot(self):
        xplot = np.arange(self.x[0], self.x[-1], self.step)

        plot()