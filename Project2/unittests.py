
"""
Project 2 FMNN25 Advanced Numerical Algorithms with Python/SciPy
Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)

"""""

import unittest
from optimizer import OptimizationProblem
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
        def grad(x):
            return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
        def hessian(x):
            return np.array([[-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]])
        self.opt = OptimizationProblem(func)
        self.func = func
        self.grad = grad
        self.hessian = hessian



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



        alpha_gold = ls.ls_gold(self.func, self.grad, x, p, alp_0, rho)
        alpha_wolfe = ls.ls_wolfe(self.func, self.grad, x, p, alp_0, rho, sigma)

        print('alpha_gold:', alpha_gold, 'alpha_wolfe:', alpha_wolfe)

        print('f:', self.func(np.array([alpha_gold, 0])),'f2:', self.func(np.array([alpha_wolfe,0])))

        #Jämför med svaret p.37
        self.assertAlmostEqual(alpha_wolfe,0.16094,delta=0.3)

    def test_newton(self):
        pass

            #Lina och Björn klistra in sina test här

    def test_invHessian(self):
        """
        Test quality of approximation of bfgs method of inverse of Hessian for growing k
        :return:
        """

        def calc_H_inv(self,x0):
            maxit = 100
            x = np.copy(x0)
            n = len(x)

            # Initial guess of inverse of Hessian
            H = np.identity(n)
            normerr = np.zeros(maxit)

            for i in range(maxit):
                # Search direction
                p = -np.dot(H, self.grad(x))

                alpha = ls.ls_exact(self.func, x, p)

                w = alpha * p
                x = x + w
                if np.linalg.norm(p) < 10 ** -6:
                    print("Converged in " + str(i) + " iterations!")
                    break

                # BFGS update of H inverse
                y = self.grad(x) - self.grad(x - w)
                rho = 1. / np.inner(y, w)

                H = np.dot(np.dot(np.identity(n) - rho * np.outer(w, y), H),
                           (np.identity(n) - rho * np.outer(y, w))) + rho * np.outer(w, w)
                normerr[i] = np.linalg.norm(H - np.linalg.inv(self.hessian(x)))
                print(H)
                print(np.linalg.inv(self.hessian(x)))

            return H, normerr, x

        x0 = np.array([0, 0])
        H_inv, normerr, x_calc = calc_H_inv(x0)
        self.assertAlmostEqual(normerr[-1], 0)





if __name__ == '__main__':
    unittest.main()