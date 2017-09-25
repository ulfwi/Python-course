
"""
Project 2 FMNN25 Advanced Numerical Algorithms with Python/SciPy
Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)

"""""

import unittest
from optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import line_search_methods as ls
import scipy.linalg as la

class TestOptimizer(unittest.TestCase):


    def setUp(self):
        """
        Creates an optimizer object with objective function func

        :return: --
        """""
        # Rosenbrock function
        def func(x):
            return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
        self.opt = Optimizer(func)
      # Måste ändra här - nu när vi har problemclass

    def tearDown(self):
        pass


    # Task 7:
    # Test this seperately from an optimization method on Rosenbrock’s function and use the parameters given on p.37 in
    # the book mentioned above.
    def test_ls(self):
        """
        Test the inexact line search method based on the Goldstein/Wolfe conditions

        :return: --
        """""

        #Parameters given on p.37
        sigma = 0.1
        rho = 0.01
        alp_0=0.1
        x = np.array([0, 0])
        p = np.array([1,0])

        # Rosenbrock function
        def func(x):
            return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

        # Grad Rosenbrock function
        def grad(x):
            return np.array([-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0]),200*(x[1]-x[0]**2)])

        alpha_gold = ls.ls_gold(func, grad, x, p, alp_0, rho)
        alpha_wolfe = ls.ls_wolfe(func, grad, x, p, alp_0, rho, sigma)

        print('alpha_gold:', alpha_gold, 'alpha_wolfe:', alpha_wolfe)

        print('f:', func(np.array([alpha_gold, 0])),'f2:', func(np.array([alpha_wolfe,0])))

        #Jämför med svaret p.37
        #self.assertAlmostEqual(alpha_wolfe,1.16094)

        def test_newton:

            #Lina och Björn klistra in sina test här

