import numpy as np

import math

class Spline:

    # u are node points
    # y are node function values
    # d are deBoor points
    # x
    # N
    def __init__(self, u, y, c, step = 0.01):
        self.u = u
        self.y = y
        self.step = step
        self.c = self.u # vad i hela världen ska denna vara?
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
    def deBoorPoints(self, u_index):
        u_short = self.u[2:-2]
        d = np.array([np.zeros(4)])
        alpha = np.array([np.zeros(4)])
        for i in range(0, np.len(d)-1):
            d[i] = self.c[u_index-3+i]
            alpha[i] = u[u_index] -
        #return d?

        # HUR VÄLJS CONTROL POINTS?? KOLLA MER PÅ DET.

    def plot(self):
        xplot = np.arange(self.x[0], self.x[-1], self.step)

        plot()