import numpy as np
import scipy.linalg as la

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Spline:

    """
    u are node points, increasing
    d are deBoor points
    N length of parameter vector
    """
    def __init__(self, coord, u, interpolate=False, N=100):
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

    def __call__(self, coord, u, interpolate=False, N=100):
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
        print("Now you've gotten a new spline (using your old one), yay!")



    def byInterpolation(self, coord):
        """
        Define control points by interpolation.

        :param coord: coordinates on curve for interpolation
        :return: interpolated control/deBoor points d
        """""

        # form chi Greville abscissae from unpadded u
        chi = (self.u[:-2] + self.u[1:-1] + self.u[2:]) / 3.
        print(chi)
        # form Vandermonde like system
        L = coord.shape[1]
        # V = np.zeros([L,L])
        # for j in range(chi.shape[0]):
        #    for i in range(L):
        #        V[j,i] = self.basis(chi[i], j)

        # FIXA
        V = np.array([np.array([self.basis(chi[i], j) for j in range(L)]) for i in range(chi.shape[0])])
        print(V)

        dx = la.solve(V, coord[0, :])  # solve banded? (2,2)
        dy = la.solve(V, coord[1, :])
        d = np.array([dx, dy])
        return d


    def basis(self, x, i, k=3):
        """
        Evaluates basis function N_i^3 in point x.

        :param x: point to be evaluated
        :param i: index of basis function
        :param k: degree of basis function
        :return: basis function i evaluated at x
        """""

        # Base case for k = 0
        if k == 0:
            if self.u[i - 1] <= x <= self.u[i]:
                return 1
            else:
                return 0

        # If at left boundary, preventing index out of bounds, no contribution from the other basis functions
        if i == 0:
            return self.div(self.basis(x, i + 1, k - 1) * (self.u[i + k] - x), (self.u[i + k] - self.u[i]))

        # If at right boundary, preventing index out of bounds, no contribution from the other basis functions
        if i == self.u.shape[0] - 3:
            return self.div(self.basis(x, i, k - 1) * (x - self.u[i - 1]), (self.u[i + k - 1] - self.u[i - 1]))

        # Recursion step
        return self.div(self.basis(x, i, k - 1) * (x - self.u[i - 1]), (self.u[i + k - 1] - self.u[i - 1])) \
               + self.div(self.basis(x, i + 1, k - 1) * (self.u[i + k] - x), (self.u[i + k] - self.u[i]))

    def spline(self):
        """
        Creates a spline.

        Keyword arguments:
        t_point -- point to be evaluated
        """""

        # Spline
        spline_vec = np.zeros((self.t.shape[0],2))
        for t_point in range(self.t.shape[0]):
            spline_vec[t_point] = self.de_Boor_points(self.t[t_point])
        return spline_vec


    def blossom(self, t_point):
        """
        Recursively evaluate the spline /deBoor algorithm.

        :param t_point: point to be evaluated, s(t_point) defined at u[2:-2]
        :return: d_fou (s(t_point)), spline evaluated at s(t_point)
        """""

        # Find hot interval
        ind = (t_point <= self.u[2:-1]).argmax() + 2  # u_I
        if ind >= (len(self.u[0:-2])-1):
            ind = len(self.u[0:-2])-2

        # Select corresponding control points d_i
        d_org = np.array([[0,0], [0,0], [0,0], [0,0]])
        for i in range(4):
            d_org[i] = np.array([self.d[0, ind - 2 + i], self.d[1, ind - 2 + i]])
        alpha = self.div(self.u[ind + 1] - t_point, self.u[ind + 1] - self.u[ind - 2])

        # Blossom recursion, sec for second interpolation etc
        d_sec = np.array([alpha * d_org[i] + (1 - alpha) * d_org[1 + i] for i in range(d_org.shape[0] - 1)])
        d_thr = np.array([alpha * d_sec[i] + (1 - alpha) * d_sec[1 + i] for i in range(d_sec.shape[0] - 1)])
        d_fou = np.array([alpha * d_thr[i] + (1 - alpha) * d_thr[1 + i] for i in range(d_thr.shape[0] - 1)])

        return d_fou  # s(t_point)


    def plotSpline(self, polygon=True):
        """
        Plots spline and polygon with marked control points.

        :param polygon: True if plotting polygon, otherwise False (boolean)
        :return: --
        """""

        # Plot polygon
        if polygon:
            plt.plot(self.d[0,:], self.d[1,:],'r-*')

        # Plot spline
        s = self.spline()
        x = np.zeros(s.shape[0])
        y = np.zeros(s.shape[0])
        for i in range(s.shape[0]):
            x[i] = s[i,0]
            y[i] = s[i,1]
        plt.plot(x,y,'b')
        plt.show()

    def plotAllBases(self):
        """
        Plots all basis functions.

        :return: --
        """""

        for i in range(self.u.shape[0] - 2):
            N = np.array([self.basis(self.t[j], i) for j in range(self.t.shape[0])])
            plt.plot(self.t, N)
        plt.title('Cubic B-spline basis functions $N_i^3$')
        plt.show()

    def plotSplineInterpol(self):
        """
        Method to calculate and plot the spline and control polygon by interpolation.

        :return: --
        """""

        L = self.d.shape[1]
        N = np.array([[self.basis(self.t[i], j) for j in range(L)] for i in range(self.t.shape[0])])
        sx = np.dot(N, self.d[0, :])
        sy = np.dot(N, self.d[1, :])
        plt.plot(sx, sy, 'b-')
        plt.plot(self.d[0, :], self.d[1, :], 'r-*')
        plt.title('Spline by basis definition')
        plt.show()

    def div(self,x,y):
        """
        Definition of zero divided by zero.

        :param x: nominator
        :param y: denominator
        :return: x divided by y
        """""
        if y == 0 and x == 0:
            return 0
        return x / y

