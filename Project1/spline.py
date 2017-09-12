import numpy as np
import scipy.linalg as linalg

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random


class Spline:

    """
    u are node points, increasing
    d are deBoor points
    N length of parameter vector
    """
    def __init__(self, d, u, interpolate=False, N=1000): # 100
        self.N = N
        self.u = u
        self.u.sort()  # if not sorted in increasing order
        # Create parameter vector t of length N
        self.t = np.arange(self.u[0], self.u[-1], self.u[-1] / N)
        # First and last three elements should be equal
        self.u = np.append([self.u[0], self.u[0]], self.u)
        self.u = np.append(self.u, [self.u[-1], self.u[-1]])
        # Control points already defined
        if interpolate == False:
            self.d = d
        # Define by interpolation
        else:
            self.d = self.byInterpolation(d)


    def __call__(self, d, u, interpolate=False, N=100):
        self.u = u
        self.u.sort()  # if not sorted in increasing order
        # Create parameter vector t of length N
        self.t = np.arange(self.u[0], self.u[-1], self.u[-1] / N)
        # First and last three elements should be equal
        self.u = np.append([self.u[0], self.u[0]], self.u)
        self.u = np.append(self.u, [self.u[-1], self.u[-1]])
        # Control points already defined
        if interpolate == False:
            self.d = d
        # Define by interpolation
        else:
            self.d = self.byInterpolation(d)
        print("Now you've gotten a new spline (using your old one), yay!")


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
        """
        Creates a spline.

        Keyword arguments:
        t_point -- point to be evaluated
        """""
        # Spline
        spline_vec = np.zeros((2, self.t.shape[0]))
        for t_point in range(self.t.shape[0]):
            blossom_pair = self.blossom(self.t[t_point])
            blossom_pair = blossom_pair[0]
            spline_vec[0, t_point] = blossom_pair[0]
            spline_vec[1, t_point] = blossom_pair[1]
        return spline_vec


    def blossom(self, t_point):
        """
        Recursively evaluate the spline /deBoor algorithm.

        :param t_point: point to be evaluated, s(t_point) defined at u[2:-2]
        :return: d_3 (s(t_point)), spline evaluated at s(t_point)
        """""
        #'''
        # Find hot interval
        ind = (t_point <= self.u[2:]).argmax() + 1  # u_I
        if ind < 2:
            ind += 1

        '''
        ind = (t_point <= self.u[2:]).argmax() + 2  # u_I
        print(ind)
        if ind > (len(self.u[0:-2])-2):
            ind -= 1
            print('inside')
        '''
        # Select corresponding control points d_i
        d_0 = np.array([[0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]])
        k = 0
        for i in range(4):
            d_0[i] = np.array([self.d[0, ind - 2 + i], self.d[1, ind - 2 + i]])

        # Blossom recursion, sec for second interpolation etc
        d_1 = np.array([(self.alpha(1, i, ind, t_point) * d_0[i] + (1 - self.alpha(1, i, ind, t_point)) * d_0[1 + i]) for i in range(d_0.shape[0] - 1)])
        d_2 = np.array([(self.alpha(2, i, ind, t_point) * d_1[i] + (1 - self.alpha(2, i, ind, t_point)) * d_1[1 + i]) for i in range(d_1.shape[0] - 1)])
        d_3 = np.array([(self.alpha(3, i, ind, t_point) * d_2[i] + (1 - self.alpha(3, i, ind, t_point)) * d_2[1 + i]) for i in range(d_2.shape[0] - 1)])

        return d_3  # s(t_point)


    def alpha(self, k, i, ind, t_point):
        '''

        :param k: k:th interpolation
        :param i:
        :param ind:
        :param t_point:
        :return:
        '''
        i = i + k - 1
        return self.div(self.u[ind + (i - 2) + (4 - k)] - t_point, self.u[ind + (i - 2) + (4 - k)]- self.u[ind + (i - 2)])


    def add(self, spline_1):
        """"
        Add two splines. This object and spline_1

        :param spline_1: spline to add
        :return: --
        """""

        # SUCCESSIVE ELEMENTS 0 0 problem+-??!? u

        # Add the knots and redefine attributes
        u_1 = spline_1.u[2:-2]
        u = np.append(u_1, self.u[2:-2])
        u.sort()
        # Find duplets, exchange to random number (non-duplet)
        u_list = u.tolist()
        u_list = list(set(u_list))
        duplets = len(u.tolist()) - len(u_list)
        duplets += 25
        while not (duplets == 0):
            rnd = random.uniform(u[0], u[-1])
            if (rnd in u_list):
                pass
            else:
                u_list.append(rnd)
                duplets -= 1
        u = np.array(u_list)
        u.sort()
        self.t = np.arange(self.u[0], self.u[-1], self.u[-1] / self.N)
        # First and last three elements should be equal
        u = np.append([u[0], u[0]], u)
        u = np.append(u, [u[-1], u[-1]])
        self.u = u
        print(self.u)
        #print(type(self.u))

    # Add the control points and redefine attributes
        d_1 = spline_1.d
        x = np.concatenate((self.d[0, :], d_1[0, :]), axis=0)
        y = np.concatenate((self.d[1, :], d_1[1, :]), axis=0)
        d = np.array([x, y])
        #d.sort()
        self.d = d
        #print(self.d)
        #print(type(self.d))
        return 0

    def __add__(self, spline_1):
        """"
        Add two splines. This object and spline_1

        :param spline_1: spline to add
        :return: --
        """""

        # SUCCESSIVE ELEMENTS 0 0 problem+-??!? u

        # Add the knots and redefine attributes
        u_1 = spline_1.u[2:-2]
        print(spline_1.u)
        print(self.u)
        u = np.append(u_1, self.u[2:-2])
        u.sort()
        # Find duplets, exchange to random number (non-duplet)
        u_list = u.tolist()
        print('u_lsit before:', len(u_list))
        print('     u_lsit before:', (u_list))
        u_list = list(set(u_list))
        duplets = len(u.tolist()) - len(u_list)
        duplets += 2 # 2 elements in u too little when adding two
        while not (duplets == 0):
            rnd = random.uniform(u[0], u[-1])
            if (rnd in u_list):
                pass
            else:
                u_list.append(rnd)
                duplets -= 1
        u = np.array(u_list)
        u.sort()
        print('u_list after:', len(u))
        print('     u_list after:', (u))

        # Add the control points and redefine attributes
        d_1 = spline_1.d
        x = np.concatenate((self.d[0, :], d_1[0, :]), axis=0)
        y = np.concatenate((self.d[1, :], d_1[1, :]), axis=0)
        d = np.array([x, y])
        print('d: ', d.shape[1])
        return Spline(d,u)




    def plotSpline(self, polygon=True):
        """
        Plots spline and polygon with marked control points

        :param polygon: True if plotting polygon, otherwise False (boolean)
        :return: --
        """""

        # Plot polygon
        if polygon:
            plt.plot(self.d[0,:], self.d[1,:],'r-*')

        # Plot spline
        s = self.spline()

        plt.plot(s[0,:], s[1,:], 'b')
        plt.show()

        return 0


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

