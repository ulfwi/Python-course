import numpy as np

import math

class Spline:


    # (self, u, y, c, step = 0.01):
    """
    u are node points
    d are deBoor points in the form np.array([[d_x,d_y], [d_x,d_y]..])
    """
    def __init__(self, N, d):
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
        for t_point in range(self.t.shape):
            spline_vec[t_point] = self.de_Boor_points(self.t[t_point], 1)
        return spline_vec

    '''
    Recursively evaluate the spline /deBoor algorithm
    t_point: is s(t_point) (u in lecture notes), s(t_point) defined at u[2:-2]
    d_choice: d given or calculated by interpolation with given (x,y)
    '''
    def de_Boor_points(self, t_point, d_choice):
        # Find hot interval
        index = (t_point < self.u).argmax()-1 # u_I
        # Select corresponding control points d_i
        if (d_choice == 1):
            d_org = self.d_given(t_point, index)
        if (d_choice == 2):
            d_org = self.d_interpolation(t_point, index)
        alpha = (self.u[index+1] - t_point)/(self.u[index+1] - self.u[index-2])

        # Blossom recursion, sec for second interpolation etc
        d_sec = [alpha*d_org[index-2+i] + (1-alpha)*d_org[index-1+i] for i in range(d_org.shape[0]-1)]
        d_thr = [alpha*d_sec[index-2+i] + (1-alpha)*d_sec[index-1+i] for i in range(d_sec.shape[0]-1)]
        d_fou = [alpha*d_thr[index-2+i] + (1-alpha)*d_thr[index-1+i] for i in range(d_thr.shape[0]-1)]

        return d_fou # s(t)

    '''
    Returns surrounding d-points for index-1 < t_point < index (??, tänk lute
    '''
    def d_given(self, index):
        # Select corresponding control points d_i
        d_org = [self.d[index-2+i] for i in range(4)]
        return d_org

    def d_interpolation(self, index):
        ...
        return d_org


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
