"""
Project 2 FMNN25 Advanced Numerical Algorithms with Python/SciPy
Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)

"""""

import scipy.linalg as la
import numpy as np
import line_search_methods as ls


class OptimizationProblem:

    def __init__(self, func, grad=None):
        """
        Constructs an Optimization Problem
        :param func: Function to be minimized
        :param grad: Gradient of the function
        """
        if grad is None:
            def grad(x):
                return self.finite_diff(func, x)
        self.func = func
        self.grad = grad

    def finite_diff(self, f, x, h=1e-8):
        """
        Approximates the gradient of the function f in a point x using finite differences.
        :param f: Function to be differentiated
        :param x: Point where the gradient is approximated
        :param h: Step length used in approximation (optional)
        :return: Gradient of function f in point x
        """
        g = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x_upper = np.copy(x)
            x_lower = np.copy(x)
            x_upper[i] += h
            x_lower[i] -= h
            g[i] = (f(x_upper) - f(x_lower)) / (2 * h)
        return g

    def calc_hessian(self, x, h = 1e-8):
        """
        Approximates the Hessian of the function in a point x using finite differences.
        :param x: Point where the gradient is approximated
        :param h: Step length used in approximation (optional)
        :return: Hessian of the function in point x
        """
        n = x.shape[0]
        hess = np.zeros([n,n])
        for k in range(n):
            x_upper = np.copy(x)
            x_lower = np.copy(x)
            x_upper[k] += h
            x_lower[k] -= h
            g_upper = self.grad(x_upper)
            g_lower = self.grad(x_lower)
            hess[k, :] = (g_upper - g_lower) / (2 * h)
        return hess

    def newton_solve(self, x0, solver = "newton", line_search = "exact", tol=10 ** -6, maxit = 1000):
        """
        Finds a minima of the function using either "Newton's method", "BFGS",
        "DFP", "good Broyden" or "bad Broyden" with either "Exact line search",
        "Wolfe conditions" or "Goldstein conditions".

        :param x0: Starting point
        :param solver: Solver to be used when finding the minima. Choose from:
        "newton", "bfgs", "dfp", "goodBroyden" and "badBroyden".
        :param line_search: Line search to be used when finding the minima. Choose from:
        "exact", "wolfe" and "goldstein".
        :param tol: Tolerance for how close to the minima we need to get
        :param maxit: Maximum number of iterations
        :return: Minimum point x
        """
        if solver == "newton":
            return self.newton_method(x0, line_search, tol, maxit)
        elif solver == "bfgs":
            return self.bfgs_method(x0, line_search, tol, maxit)
        elif solver == "dfp":
            return self.dfp_method(x0, line_search,tol, maxit)
        elif solver == "goodBroyden":
            return self.good_broyden_method(x0, line_search,tol, maxit)
        elif solver == "badBroyden":
            return self.bad_broyden_method(x0, line_search,tol, maxit)

    def newton_method(self, x0, line_search, tol = 10 ** -6, maxit = 1000):
        """
        Finds the minima using Newton's method
        :param x0: Starting point
        :param line_search: Line search method to be used
        :param tol: Tolerance for how close to the minima we need to get
        :param maxit: Maximum number of iterations
        :return: Minimum point x
        """
        x = np.copy(x0)

        for i in range(maxit):

            # Approximate Hessian by finite differences
            G = self.calc_hessian(x)
            G = 0.5*(np.conjugate(G) + np.transpose(np.conjugate(G)))
            try:
                L = la.cholesky(G)
                y = la.solve(np.transpose(L),self.grad(x))
                p = -la.solve(L, y)
            except Exception:
                #print("Hessian not spd! Solving linear system without Cholesky factorization.")
                p = -la.solve(G, self.grad(x))
            if line_search == "exact":
                alpha = ls.ls_exact(self.func, x, p)
            elif line_search == "goldstein":
                alpha = ls.ls_gold(self.func, self.grad, x, p, tol)
            elif line_search == "wolfe":
                alpha = ls.ls_wolfe(self.func, self.grad, x, p, tol)

            x = x + alpha*p
            if la.norm(p) < tol:
                print("Converged in " + str(i) + " iteration(s)!")
                return x
           # print("Iteration: " + str(i) + " Step: " + str(p))
        print("Did not converge. Number of iterations: " + str(i) + "\nFinal error: " + str(la.norm(p)))
        return 1

    def bfgs_method(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        """
        Finds the minima using BFGS method
        :param x0: Starting point
        :param line_search: Line search method to be used
        :param tol: Tolerance for how close to the minima we need to get
        :param maxit: Maximum number of iterations
        :return: Minimum point x
        """
        x = np.copy(x0)
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot(H,self.grad(x))
            if line_search == "exact":
                alpha = ls.ls_exact(self.func, x, p)
            elif line_search == "goldstein":
                alpha = ls.ls_gold(self.func, self.grad, x, p, tol)
            elif line_search == "wolfe":
                alpha = ls.ls_wolfe(self.func, self.grad, x, p, tol)
            w = alpha*p
            x = x + w
            if la.norm(w) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x
            # BFGS update of H inverse
            y = self.grad(x) - self.grad(x-w)
            rho = 1. / np.inner(y,w)
            H = np.dot(np.dot(np.identity(n) - rho*np.outer(w,y),H), (np.identity(n) - rho*np.outer(y,w))) + rho*np.outer(w,w)
            print(H)

        print('Did not converge. Number of iterations: ' + str(i) + '\nFinal error: ' + str(la.norm(w)))

    def dfp_method(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        """
        Finds the minima using DFP method
        :param x0: Starting point
        :param line_search: Line search method to be used
        :param tol: Tolerance for how close to the minima we need to get
        :param maxit: Maximum number of iterations
        :return: Minimum point x
        """
        x = x0
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot( H, self.grad(x))
            if line_search == "exact":
                alpha = ls.ls_exact(self.func, x, p)
            elif line_search == "goldstein":
                alpha = ls.ls_gold(self.func, self.grad, x, p, tol)
            elif line_search == "wolfe":
                alpha = ls.ls_wolfe(self.func, self.grad, x, p, tol)

            s = alpha * p
            x = x + s
            if la.norm(s) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x
            y = self.grad(x) - self.grad(x - s)
            H = H - np.outer(np.dot(H,y),np.dot(np.transpose(y),H)) / np.inner(np.transpose(y),np.dot(H,y)) + np.outer(s,s) / np.inner(y,s)

        print('Did not converge. Number of iterations: ' + str(i) + '\nFinal error: ' + str(np.norm(s)))

    def good_broyden_method(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        """
        Finds the minima using good Broyden method
        :param x0: Starting point
        :param line_search: Line search method to be used
        :param tol: Tolerance for how close to the minima we need to get
        :param maxit: Maximum number of iterations
        :return: Minimum point x
        """
        x = np.copy(x0)
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot(H, self.grad(x))
            if line_search == "exact":
                alpha = ls.ls_exact(self.func, x, p)
            elif line_search == "goldstein":
                alpha = ls.ls_gold(self.func, self.grad, x, p, tol)
            elif line_search == "wolfe":
                alpha = ls.ls_wolfe(self.func, self.grad, x, p, tol)

            delta = alpha * p
            x = x + delta
            if la.norm(p) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x

            # Broyden update of H
            gamma = self.grad(x) - self.grad(x-delta)
            # u = delta - np.dot(H, gamma)
            # a = 1 / np.inner(u, gamma)
            # H = H + a * np.outer(u, u)
            # a = (delta - np.dot(H, gamma)) / np.inner(delta, np.dot(H, gamma))
            # b = np.dot(np.transpose(delta), H)
            # H = H + np.outer(a,b)

            a = (delta - np.dot(H, gamma))
            b = np.dot(np.transpose(delta),H)
            c = np.inner(delta, np.dot(H, gamma))
            H = H + np.outer(a,b)/c
            print(x)
            print(H)
            #a = np.outer(delta - np.dot(H, gamma),delta) / np.inner(delta, np.dot(H, gamma))

        print('Did not converge. Number of iterations: ' + str(i) + '\nFinal error: ' + str(la.norm(delta)))

    def bad_broyden_method(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        """
        Finds the minima using bad Broyden method
        :param x0: Starting point
        :param line_search: Line search method to be used
        :param tol: Tolerance for how close to the minima we need to get
        :param maxit: Maximum number of iterations
        :return: Minimum point x
        """
        x = np.copy(x0)
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot(H, self.grad(x))
            if line_search == "exact":
                alpha = ls.ls_exact(self.func, x, p)
            elif line_search == "goldstein":
                alpha = ls.ls_gold(self.func, self.grad, x, p, tol)
            elif line_search == "wolfe":
                alpha = ls.ls_wolfe(self.func, self.grad, x, p, tol)

            delta = alpha * p
            x = x + delta
            if la.norm(p) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x

            # Broyden update of H
            gamma = self.grad(x) - self.grad(x-delta)
            H = H + np.outer((delta - np.dot(H,gamma)) / np.inner(gamma,gamma), delta)

        print('Did not converge. Number of iterations: ' + str(i) + '\nFinal error: ' + str(la.norm(p)))


