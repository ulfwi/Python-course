
import numpy.linalg as la
import numpy as np
import scipy.optimize as op

class Optimizer:

    def __init__(self, func, grad = None):
        if grad is None:
            def grad(x):
                return op.approx_fprime(x, func, 10**-6)
        self.grad = grad
        self.func = func


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




            # Search direction
            L = la.cholesky(G)
            p = -np.solve(L*np.transpose(L),self.grad)
            if line_search == "exact":
                alpha = self.ls_exact(p,x,tol)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p,x,tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            x = x + alpha*p
            if np.norm(p) < tol:
                print('Converged in' + i + 'iterations!')
                return x
        print('Did not converge. Number of iterations: ' + maxit + '\nFinal error: ' + np.norm(p))

    def newton_solve(self, x0, line_search, tol = 10 ** -6, maxit = 1000):
        x = x0
        for i in range(maxit):
            G = op.approx_fprime(x, self.grad,tol)
            G = 0.5*(np.conjugate(G) + np.transpose(np.conjugate(G)))
            try:
                L = la.cholesky(G)
            except Exception:
                pass

            p = -np.solve(L * np.transpose(L), self.grad)
            if line_search == "exact":
                alpha = self.ls_exact(p, x, tol)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p, x, tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            x = x + alpha * p
            if np.norm(p) < tol:
                print('Converged in' + i + 'iterations!')
                return x
        print('Did not converge. Number of iterations: ' + maxit + '\nFinal error: ' + np.norm(p))


    def bfgs_solve(self, x0, line_search, tol=10 ** -6, maxit = 1000):
        x = x0
        n = len(x)
        # Initial guess of inverse of Hessian
        H = np.identity(n)

        for i in range(maxit):
            # Search direction
            p = -H*self.grad(x)
            if line_search == "exact":
                alpha = self.ls_exact(p, x, tol)
            elif line_search == "goldstein":
                alpha = self.ls_gold(p, x, tol)
            elif line_search == "wolfe":
                alpha = self.ls_gold(p, x, tol)

            w = alpha*p
            x = x + w
            if np.norm(w) < tol:
                print('Converged in' + i + 'iterations!')
                return x
            y = self.grad(x) - self.grad(x-w)
            rho = 1 / np.dot(y,w)
            H = (np.identity(n) - rho*np.outer(w,y))*H*(np.identity - rho*np.outer(y,w)) + rho*np.outer(w,w)

        print('Did not converge. Number of iterations: ' + maxit + '\nFinal error: ' + np.norm(w))




    def secant_method_solve(self, line_search, tol=10 ** -6, maxit = 1000):
        pass


    def ls_exact(self, p, x, tol = 10 ** -6):
        return 1

    def ls_gold(self, p, x, tol = 10 ** -6):
        pass

    def ls_wolfe(self, p, x, tol = 10 ** -6):
        pass