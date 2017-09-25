
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

        def test_newton(self):
            """
            Finds the minimum of the Rosenbrock function with Newton's method with different
            line search methods and compares them with the analytical solution
            :return:
            """
            tol = 10 ** -6
            x_opt_exact = self.opt.newton_solve(np.array([0., 0.]), "newton", "exact", tol, 1000)
            x_opt_wolfe = self.opt.newton_solve(np.array([0., 0.]), "newton", "wolfe", tol, 1000)
            x_opt_gold = self.opt.newton_solve(np.array([0., 0.]), "newton", "goldstein", tol, 1000)

            np.testing.assert_almost_equal(x_opt_exact, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_wolfe, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_gold, np.array([1., 1.]), 6)

        def test_bfgs(self):
            """
            Finds the minimum of the Rosenbrock function with the BFGS method with different
            line search methods and compares them with the analytical solution
            :return:
            """
            tol = 10 ** -6
            x_opt_exact = self.opt.newton_solve(np.array([0., 0.]), "bfgs", "exact", tol, 1000)
            x_opt_wolfe = self.opt.newton_solve(np.array([0., 0.]), "bfgs", "wolfe", tol, 1000)
            x_opt_gold = self.opt.newton_solve(np.array([0., 0.]), "bfgs", "goldstein", tol, 1000)

            np.testing.assert_almost_equal(x_opt_exact, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_wolfe, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_gold, np.array([1., 1.]), 6)

        def test_dfp(self):
            """
            Finds the minimum of the Rosenbrock function with DFP method with different
            line search methods and compares them with the analytical solution
            :return:
            """
            tol = 10 ** -6
            x_opt_exact = self.opt.newton_solve(np.array([0., 0.]), "dfp", "exact", tol, 1000)
            x_opt_wolfe = self.opt.newton_solve(np.array([0., 0.]), "dfp", "wolfe", tol, 1000)
            x_opt_gold = self.opt.newton_solve(np.array([0., 0.]), "dfp", "goldstein", tol, 1000)

            np.testing.assert_almost_equal(x_opt_exact, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_wolfe, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_gold, np.array([1., 1.]), 6)

        def test_good_broyden(self):
            """
            Finds the minimum of the Rosenbrock function with good Broyden method with different
            line search methods and compares them with the analytical solution
            :return:
            """
            tol = 10 ** -6
            x_opt_exact = self.opt.newton_solve(np.array([0., 0.]), "goodBroyden", "exact", tol, 1000)
            x_opt_wolfe = self.opt.newton_solve(np.array([0., 0.]), "goodBroyden", "wolfe", tol, 1000)
            x_opt_gold = self.opt.newton_solve(np.array([0., 0.]), "goodBroyden", "goldstein", tol, 1000)

            np.testing.assert_almost_equal(x_opt_exact, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_wolfe, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_gold, np.array([1., 1.]), 6)

        def test_bad_broyden(self):
            """
            Finds the minimum of the Rosenbrock function with bad Broyden method with different
            line search methods and compares them with the analytical solution
            :return:
            """
            tol = 10 ** -6
            x_opt_exact = self.opt.newton_solve(np.array([0., 0.]), "badBroyden", "exact", tol, 1000)
            x_opt_wolfe = self.opt.newton_solve(np.array([0., 0.]), "badBroyden", "wolfe", tol, 1000)
            x_opt_gold = self.opt.newton_solve(np.array([0., 0.]), "badBroyden", "goldstein", tol, 1000)

            np.testing.assert_almost_equal(x_opt_exact, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_wolfe, np.array([1., 1.]), 6)
            np.testing.assert_almost_equal(x_opt_gold, np.array([1., 1.]), 6)

