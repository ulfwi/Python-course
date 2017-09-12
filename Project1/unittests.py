import unittest
from spline import Spline
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class TestSpline(unittest.TestCase):

    def setUp(self):
        """
        Creates a spline with uniform grid.
        
        :return: -- 
        """""
        n = 10 # number of points
        x = np.linspace(0, 2 * np.pi, n) #test with sine-curve
        y = np.sin(x)
        self.u = np.arange(len(x) - 2)
        coord = np.array([x, y])
        self.sp = Spline(coord,self.u,True) # define from interpolation

    def tearDown(self):
        pass


    def test_normalized(self):
        """
        Test that asserts sum(Ni(u)) = 1 for any u in [u_2, u_{K-2}]
        
        :return: --
        """""

        t = self.u[0] + (self.u[-1]-self.u[0])*np.random.rand(1000) #randomized values over interval
        np.append(t,self.u[-1]) #include final value
        for i in range(t.shape[0]):
            N = np.array([self.sp.basis(t[i], j) for j in range(self.u.shape[0]+2)])
            self.assertAlmostEqual(1., np.sum(N))

    def test_byInterpolation(self):
        """
        Test how well sine-function can be reproduced using spline interpolation.
         
        :return: --
        """""
        spline_vec = self.sp.spline()
        np.testing.assert_almost_equal(spline_vec[1,:], np.sin(spline_vec[0,:]),1)

    def test_blossom(self):
        """
        Test that spline calculation using basis interpolation and Blossom algorithm gives the same result,
        s(u) = sum(d*N).
        
        :return: -- 
        """""
        s_vec_blossom = self.sp.spline()
        s_vec_intpol = self.sp.splineInterpol()
        np.testing.assert_almost_equal(s_vec_blossom, s_vec_intpol)

    def test_runtimes(self):
        """
        Check runtimes of blossom algorithm vs basis interpolation.
        
        :return: --
        """""

        tmin_intpol = 100
        tmin_blossom = 100

        for i in range(10):
            start = timer()
            s_vec_intpol = self.sp.splineInterpol()
            end = timer()
            if (end - start < tmin_intpol):
                tmin_intpol = end - start
            start = timer()
            s_vec_blossom = self.sp.spline()
            end = timer()
            if (end - start < tmin_intpol):
                tmin_blossom = end - start

        print('Runtime basis interpolation: ',tmin_intpol)
        print('Runtime blossom algorithm: ', tmin_blossom)

    def test_uremark(self):
        """
        Test that contribution from basis function N_0^2 that is multiplied with u[-1]-term is always zero.
        
        :return: --
        """""
        t = self.u[0] + (self.u[-1] - self.u[0]) * np.random.rand(100)  # randomized values over interval
        np.append(t, self.u[-1])  # include final value
        N = self.sp.makeBasisFunc(0,2) # create function N_0^2
        Nvec = np.array([N(t[i]) for i in range(t.shape[0])])
        self.assertAlmostEqual(0, np.linalg.norm(Nvec))



if __name__ == '__main__':
    unittest.main()