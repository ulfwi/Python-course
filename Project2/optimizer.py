
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
        elif solver == "secant":
            return self.secant_method_solve(x0, line_search,tol, maxit)

    def newton_solve(self, x0, line_search, tol = 10 ** -6, maxit = 1000, h = 0.001):
        x = np.copy(x0)

        for i in range(maxit):

            # Approximate Hessian by finite differences
            G = self.calc_hessian(x,h)
            G = 0.5*(np.conjugate(G) + np.transpose(np.conjugate(G)))
            try:
                L = la.cholesky(G)
                p = -la.solve(L * np.transpose(L), self.grad(x))
            except Exception:
                print("Hessian not spd! Solving linear system without Cholesky factorization.")
                p = -la.solve(G, self.grad(x))


            if line_search == "exact":
                alpha = self.ls_exact(p, x, tol)
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
                alpha = self.ls_exact(p, x, tol)
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




    def secant_method_solve(self, line_search, tol=10 ** -6, maxit = 1000):
        pass

    def dfp_solve(self, line_search, tol=10 ** -6, maxit = 1000):
        pass


    def ls_exact(self, p, x, tol = 10 ** -6):
        return 1

    def ls_gold(self, p, x, tol = 10 ** -6):
        pass

    def ls_wolfe(self, p, x, tol = 10 ** -6):
        pass