"""
Project 2 FMNN25 Advanced Numerical Algorithms with Python/SciPy
Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)

"""""

import scipy.linalg as la
import numpy as np
import line_search_methods as ls


class OptimizationProblem:

    def __init__(self, func, grad=None):
        if grad is None:
            def grad(x):
                return self.finite_diff(func, x)
        self.func = func
        self.grad = grad

    def finite_diff(self, f, x, h=1e-8):
        g = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x_upper = np.copy(x)
            x_lower = np.copy(x)
            x_upper[i] += h
            x_lower[i] -= h
            g[i] = (f(x_upper) - f(x_lower)) / (2 * h)
        return g

    def newton_solve(self, x0, solver = "finiteDiff", line_search = "exact", tol=10 ** -6, maxit = 1000):
        if solver == "finiteDiff":
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

    def calc_hessian(self, x, h = 1e-8):
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

    def bfgs_method(self, x0, line_search, tol=10 ** -6, maxit = 1000):
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


