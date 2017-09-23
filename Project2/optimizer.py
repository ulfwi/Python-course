"""
Project 2 FMNN25 Advanced Numerical Algorithms with Python/SciPy
Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)

"""""

import scipy.linalg as la
import numpy as np
import scipy.optimize as op


class Optimizer:

    def __init__(self, func, grad = None):
        if grad is None:
            def grad(x):
                return self.finite_diff(func, x)
        self.grad = grad
        self.func = func

    def finite_diff(self, f, x, h=1e-8):
        g = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x_upper = np.copy(x)
            x_lower = np.copy(x)
            x_upper[i] += h
            x_lower[i] -= h
            g[i] = (f(x_upper) - f(x_lower)) / (2 * h)
        return g

    @classmethod
    def optimize(cls, func, grad = None, tol = 10**-6):
        pass



    def newton_method(self, x0, solver = "finiteDiff", line_search = "exact", tol=10 ** -6, maxit = 1000):
        if solver == "finiteDiff":
            return self.newton_solve(x0, line_search, tol, maxit)
        elif solver == "bfgs":
            return self.bfgs_solve(x0, line_search, tol, maxit)
        elif solver == "dfp":
            return self.dfp_solve(x0, line_search,tol, maxit)
        elif solver == "goodBroyden":
            return self.good_broyden_solve(x0, line_search,tol, maxit)
        elif solver == "badBroyden":
            return self.bad_broyden_solve(x0, line_search,tol, maxit)

    def newton_solve(self, x0, line_search, tol = 10 ** -6, maxit = 1000, h = 0.001):
        x = np.copy(x0)

        for i in range(maxit):

            # Approximate Hessian by finite differences
            G = self.calc_hessian(x,h)
            G = 0.5*(np.conjugate(G) + np.transpose(np.conjugate(G)))
            try:
                L = la.cholesky(G)
                p = -la.solve(L * np.transpose(L), self.grad(x))
                print(p)
            except Exception:
                print("Hessian not spd! Solving linear system without Cholesky factorization.")
                p = -la.solve(G, self.grad(x))

            if line_search == "exact":
                alpha = self.ls_exact(p, x)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p, x, tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            x = x + alpha*p
            if la.norm(p) < tol:
                print("Converged in " + str(i) + " iteration(s)!")
                return x
            print("Iteration: " + str(i) + " Step: " + str(p))
        print("Did not converge. Number of iterations: " + str(maxit) + "\nFinal error: " + str(la.norm(p)))
        return 1

    def calc_hessian(self, x, h = 1e-8):
        n = x.shape[0]
        G = np.zeros([n,n])
        for k in range(n):
            x_upper = np.copy(x)
            x_lower = np.copy(x)
            x_upper[k] += h
            x_lower[k] -= h
            g_upper = self.grad(x_upper)
            g_lower = self.grad(x_lower)
            G[:, k] = (g_upper - g_lower) / (2 * h)
        return G

    def bfgs_solve(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        x = np.copy(x0)
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot(H,self.grad(x))
            if line_search == "exact":
                alpha = self.ls_exact(p, x)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p, x, tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            w = alpha*p
            x = x + w
            if la.norm(w) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x
            # BFGS update of H inverse
            y = self.grad(x) - self.grad(x-w)
            rho = 1. / np.dot(y,w)
            H = (np.identity(n) - rho*np.outer(w,y))*H*(np.identity(n) - rho*np.outer(y,w)) + rho*np.outer(w,w)

        print('Did not converge. Number of iterations: ' + str(maxit) + '\nFinal error: ' + str(la.norm(w)))

    def dfp_solve(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        x = x0
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot( H, self.grad(x))
            if line_search == "exact":
                alpha = self.ls_exact(p, x)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p, x, tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            s = alpha * p
            x = x + s
            if la.norm(s) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x
            y = self.grad(x) - self.grad(x - s)
            H = H - np.outer(np.dot(H,y),np.dot(np.transpose(y),H)) / np.inner(np.transpose(y),np.dot(H,y)) + np.outer(s,s) / np.inner(y,s)

        print('Did not converge. Number of iterations: ' + maxit + '\nFinal error: ' + np.norm(s))

    def good_broyden_solve(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        x = np.copy(x0)
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot(H, self.grad(x))
            if line_search == "exact":
                alpha = self.ls_exact(p, x)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p, x, tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            delta = alpha * p
            x = x + delta
            print(p)
            if la.norm(p) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x

            # Broyden update of H
            gamma = self.grad(x) - self.grad(x-delta)
            u = delta - np.dot(H,gamma)
            a = 1 / np.dot(u, gamma)
            H = H + a*np.dot(u,u)

        print('Did not converge. Number of iterations: ' + str(maxit) + '\nFinal error: ' + str(la.norm(w)))

    def bad_broyden_solve(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        x = np.copy(x0)
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -np.dot(H, self.grad(x))
            if line_search == "exact":
                alpha = self.ls_exact(p, x)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p, x, tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            delta = alpha * p
            x = x + delta
            print(p)
            if la.norm(p) < tol:
                print('Converged in ' + str(i) + ' iteration(s)!')
                return x

            # Broyden update of H
            gamma = self.grad(x) - self.grad(x-delta)
            u = delta - np.dot(H,gamma)
            a = 1 / np.dot(u, gamma)
            H = H + np.outer((delta - np.dot(H,gamma)) / np.inner(delta, np.dot(H, gamma)), np.dot(delta,H))

        print('Did not converge. Number of iterations: ' + str(maxit) + '\nFinal error: ' + str(la.norm(w)))


# Lägga alla LS metoder i en egen klass?? Samma med Newton-metoderna? <-------------

    def ls_exact(self, p, x):
        """
        Exact line search. Minimize objective function func with respect to alpha (alp)
        using scikit.optimize.fmin

        :param p: direction of steepest descent
        :param x: evaluation point (??????????????)
        :return: step length alpha (alp)
        """
        alp = 0

        def h(alp, p, x):
            return self.func(x + alp*p)
        alp = op.fmin(h, alp, args= (p, x,))
        return alp

    def ls_block1(self, alp_0, alp_l, tau=0.1, chi=9):
        """
        Block 1 in algorithm for inexact line search (see p. 48 in lecture notes)

        :param alp_0: soon to be acceptable point (when rc and lc are true)
        :param alp_l: lower bound of alpha interval
        :param tau: parameter in algorithm
        :param chi: parameter in algorithm
        :return: updated alp_0 and alp_l
        """
        # Extrapolate
        d_alp_0 = (alp_0 - alp_l)*(self.grad(alp_0))/ (self.grad(alp_l) - self.grad(alp_0))
        d_alp_0 = max([d_alp_0, tau*(alp_0-alp_l)])
        d_alp_0 = min([d_alp_0, chi*(alp_0-alp_l)])
        alp_l = alp_0
        alp_0 = alp_0 + d_alp_0

        return alp_0, alp_l

    def ls_block2(self, alp_0, alp_l, alp_u, p, tau=0.1):
        """
         Block 2 in algorithm for inexact line search (see p. 48 in lecture notes)

        :param alp_0: soon to be acceptable point (when rc and lc are true)
        :param alp_u: upper bound of alpha interval
        :param alp_l: lower bound of alpha interval
        :param p: direction of steepest descent
        :param tau: parameter in algorithm
        :param chi: parameter in algorithm
        :return: updated alp_0 and alp_u
        """
        alp_u = min([alp_0, alp_u])
        # Interpolate
        alp_0_bar = ((alp_0 - alp_l)**2*self.grad(alp_l)*p)/ 2*(self.func(alp_l) - self.func(alp_0)
                                                                + (alp_0 - alp_l)*self.grad(alp_l)*p)
        alp_0_bar = max([alp_0_bar, alp_l + tau*(alp_u - alp_l)])
        alp_0_bar = min([alp_0_bar, alp_u - tau*(alp_u - alp_l)])
        alp_0 = alp_0_bar

        return alp_0, alp_u

    def ls_gold(self, p, alp_0=1, rho=0.25):
        """
        Line search with Goldstein conditions

        :param p: direction of steepest descent
        :param alp_0: soon to be acceptable point (when rc and lc are true)
        :param rho: parameter in algorithm
        :return: step length alp_0, acceptable point having rc and lc true
        """
        # Initialize upper and lower bound of alpha (alp)
        alp_l = 0
        alp_u = 10**99

        # Check if rho is between [0, 0.5]
        if rho > 0.5 or rho < 0:
            rho = 0.1
            print('Your choice of rho was out of bounds (allowed bounds [0, 0.5]), it has now been set to 0.1')

        # Check if lc (left condition) and rc (right condition) is true or false
        lc = (self.func(alp_0) >= self.func(alp_l) + (1-rho)*(alp_0-alp_u)*self.grad(alp_l)*p)  # self.grad*p = dfunc/dalp
        rc = (self.func(alp_0) <= self.func(alp_l) + rho*(alp_0-alp_u)*self.grad(alp_l)*p)

        # While lc or rc is false, update alp_l and alp_0 or alp_u and alp_0 respectively
        while ~(lc & rc):
            if ~lc:
                alp_0, alp_l = self.ls_block1(alp_0,alp_l)
            # if ~rc:
            else:
                alp_0, alp_u = self.ls_block2(alp_0, alp_l, alp_u, p)
            lc = (self.func(alp_0) >= self.func(alp_l) + (1 - rho)*(alp_0 - alp_u)*self.grad(alp_l)*p)
            rc = (self.func(alp_0) <= self.func(alp_l) + rho*(alp_0 - alp_u)*self.grad(alp_l)*p)  # self.grad*p = dfunc/dalp

        return alp_0

    def ls_wolfe(self, p, alp_0=1, rho=0.1, sigma=0.7):
        """
        Line search with Wolfe-Powell conditions

        :param p: direction of steepest descent
        :param alp_0: soon to be acceptable point (when rc and lc are true)
        :param rho: parameter in algorithm
        :param sigma: parameter in algorithm
        :return: step length alp_0, acceptable point having rc and lc true
        """
        # Initialize upper and lower bound of alpha (alp)
        alp_l = 0
        alp_u = 10 ** 99

        # Check if rho is between [0, 0.5]
        if rho > 0.5 or rho < 0:
            rho = 0.1
            print('Your choice of rho was out of bounds (allowed bounds [0, 0.5]), it has now been set to 0.1')

        # Check if sigma is bigger than rho
        if sigma > 1 or sigma < 0:
            sigma = 0.7
            print('Your choice of sigma was out of bounds (allowed bounds [0, 1], it is now set to 0.7')
        if sigma <= rho:
            sigma = 0.7
            print('Your choice of sigma was inaccurate since sigma > rho, it has now been set to 0.7')

        # Check if lc (left condition) and rc (right condition) is true or false
        lc = (self.grad(alp_0)*p >= sigma*self.grad(alp_l)*p)  # self.grad*p = dfunc/dalp
        rc = (self.func(alp_0) <= self.func(alp_l) + rho*(alp_0 - alp_u)*self.grad(alp_l)*p)

        # While lc or rc is false, update alp_l and alp_0 or alp_u and alp_0 respectively
        while ~(lc & rc):
            if ~lc:
                alp_0, alp_l = self.ls_block1(alp_0,alp_l)
            # if ~rc:
            else:
                alp_0, alp_u = self.ls_block2(alp_0, alp_l, alp_u, p)
            lc = (self.func(alp_0) >= self.func(alp_l) + (1 - rho)*(alp_0 - alp_u)*self.grad(alp_l)*p)
            rc = (self.func(alp_0) <= self.func(alp_l) + rho*(alp_0 - alp_u)*self.grad(alp_l)*p)  # self.grad*p = df/dalp

        return alp_0
