import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg

import math

class Spline:

    """
    u are node points, increasing
    d are deBoor points
    N length of parameter vector
    """
    def __init__(self, coord, u, interpolate=False, N=4): # 100
        self.u = u
        self.u.sort()  # if not sorted in increasing order
        # Create parameter vector t of length N
        self.t = np.arange(self.u[0], self.u[-1], self.u[-1] / N)
        # First and last three elements should be equal
        self.u = np.append([self.u[0], self.u[0]], self.u)
        self.u = np.append(self.u, [self.u[-1], self.u[-1]])
        # Control points already defined
        if interpolate == False:
            self.d = coord
        # Define by interpolation
        else:
            self.d = self.byInterpolation(coord)


    # Define control points by interpolation
    def byInterpolation(self, coord):
        # form chi Greville abscissae
        print(self.u[:-2])
        chi = (self.u[:-2] + self.u[1:-1] + self.u[2:])/3
        # form Vandermonde like system
        L = coord.shape[1]
        print(L)
        print(chi[2])
        # for indices I-1 to I+2? [i-1,i,i+1,i+2] range 2 to L-1
        V = np.array([[self.basis(chi[i], j) for j in range(L+1)] for i in range(chi.shape[0]+1)])
        print(V)
        dx = linalg.solve_banded((2,2),V,coord[0,:])
        dy = linalg.solve_banded((2,2),V,coord[1,:])
        d = np.array([dx,dy])
        print(d)
        return d

    # Returns basis function N_i^3 value in x
    def basis(self, x, i, k=3):
        # check if nodes coincide. use 0/0=0
        if i <= 0 or i>=self.u.shape[0]-1:
            return 0
        if k == 0:
            if self.u[i-1] <= x < self.u[i]:
                return 0.5 * (np.sign(x-self.u[i-1]) - np.sign(x-self.u[i])) # sum of two step functions
            else:
                return 0
        else:
            return self.div((x - self.u[i-1]),(self.u[i+k-1] - self.u[i-1])) * self.basis(x, i, k-1) \
                    + self.div((self.u[i+k] - x),(self.u[i+k] - self.u[i])) * self.basis(x, i+1, k-1)

    def spline(self):
        spline_vec = np.zeros((self.t.shape[0],2))
        for t_point in range(self.t.shape[0]):
            spline_vec[t_point] = self.de_Boor_points(self.t[t_point])
        return spline_vec

    '''
    Recursively evaluate the spline /deBoor algorithm name Blossom algorithm?
    t_point: is s(t_point) (u in lecture notes), s(t_point) defined at u[2:-2]
    d_choice: d given or calculated by interpolation with given (x,y)
    '''

    def de_Boor_points(self, t_point):
        # Find hot interval
        # ind = (t_point <= self.u).argmax() - 1  # u_I
        ind = (t_point <= self.u[2:-1]).argmax() + 2  # u_I
        if ind < 0:
            ind = 0
        # Select corresponding control points d_i
        d_org = np.array([self.d[ind - 2 + i] for i in range(4)])
        alpha = self.div(self.u[ind + 1] - t_point, self.u[ind + 1] - self.u[ind - 2]) # ALPHA SHOULD BE UPDATED, DEPENDS ON BLOSSOM PAIR

        # Blossom recursion, sec for second interpolation etc
        d_sec = np.array([alpha * d_org[i] + (1 - alpha) * d_org[1 + i] for i in range(d_org.shape[0] - 1)])
        d_thr = np.array([alpha * d_sec[i] + (1 - alpha) * d_sec[1 + i] for i in range(d_sec.shape[0] - 1)])
        d_fou = np.array([alpha * d_thr[i] + (1 - alpha) * d_thr[1 + i] for i in range(d_thr.shape[0] - 1)])

        return d_fou  # s(t)

 #   d_sec = np.array([alpha * d_org[ind - 2 + i] + (1 - alpha) * d_org[ind - 1 + i] for i in range(d_org.shape[0] - 1)])
 #   d_thr = np.array([alpha * d_sec[ind - 2 + i] + (1 - alpha) * d_sec[ind - 1 + i] for i in range(d_sec.shape[0] - 1)])
 #   d_fou = np.array([alpha * d_thr[ind - 2 + i] + (1 - alpha) * d_thr[ind - 1 + i] for i in range(d_thr.shape[0] - 1)])



    # polygon: boolean, plot if true
    def plotSpline(self, polygon=False):
        if polygon:
            plt.plot(self.d[0,:], self.d[1,:],'r-*')
            print(self.d[0,:])
        s = self.spline()
        print(s)
        print(s[0,:])
        for i in range(s.shape[0]):
            print(s[i,0])
            plt.plot(s[i,0],s[i,1],'b')
        print("afterafter")
        #plt.show()

        fig = plt.gcf()
        plotly_fig = tls.mpl_to_plotly(fig)
        plotly_fig['layout']['showlegend'] = True
        print("?")



    def div(self,x,y):
        if y == 0 and x == 0:
            return 0
        return x / y









