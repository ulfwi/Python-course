
"""
Project 2 FMNN25 Advanced Numerical Algorithms with Python/SciPy
Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)

"""""

import unittest
from optimizer import OptimizationProblem
import numpy as np
import line_search_methods as ls


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

    def test_ls(self):
        """
        Test the inexact line search method based on the Goldstein/Wolfe conditions using parameters given on p.37
        in Fletcher. Compare the result on p.38 with Wolfe.

        :return: --
        """""

        # Parameters given on p.37
        sigma = 0.1
        rho = 0.01
        alp_0=0.1
        x = np.array([0, 0])
        p = np.array([1,0])

        alpha_wolfe = ls.ls_wolfe(self.func, self.grad, x, p, alp_0, rho, sigma)

        # Compare with results p.37
        self.assertAlmostEqual(alpha_wolfe,0.16094,delta=0.3)

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

    # def test_good_broyden(self):
    #     """
    #     Finds the minimum of the Rosenbrock function with good Broyden method with different
    #     line search methods and compares them with the analytical solution
    #     :return:
    #     """
    #     tol = 10 ** -6
    #     x_opt_exact = self.opt.newton_solve(np.array([0., 0.]), "goodBroyden", "exact", tol, 1000)
    #     x_opt_wolfe = self.opt.newton_solve(np.array([0., 0.]), "goodBroyden", "wolfe", tol, 1000)
    #     x_opt_gold = self.opt.newton_solve(np.array([0., 0.]), "goodBroyden", "goldstein", tol, 1000)
    #
    #     np.testing.assert_almost_equal(x_opt_exact, np.array([1., 1.]), 6)
    #     np.testing.assert_almost_equal(x_opt_wolfe, np.array([1., 1.]), 6)
    #     np.testing.assert_almost_equal(x_opt_gold, np.array([1., 1.]), 6)

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

    def test_invHessian(self):
        """
        Test the quality of the approximation of the inverse of Hessian for growing k using bfgs method
        :return:
        """

        def calc_H_inv(x0):
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

            return H, normerr, x

        x0 = np.array([0, 0])
        H_inv, normerr, x_calc = calc_H_inv(x0)
        self.assertAlmostEqual(normerr[-1], 0)


if __name__ == '__main__':
    unittest.main()