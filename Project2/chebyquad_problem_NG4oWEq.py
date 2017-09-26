"""
Chebyquad Testproblem

Course material for the course FMNN25
Version for Python 3.4
Claus Führer (2016)

Edited by: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)
(Task 11)

"""

from  scipy import dot,linspace
import scipy.optimize as so
from numpy import array
import numpy as np
from optimizer import OptimizationProblem


def T(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the first kind
    x evaluation point (scalar)
    n degree 
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2. * x * T(x, n - 1) - T(x, n - 2)

def U(x, n):
    """
    Recursive evaluation of the Chebychev Polynomials of the second kind
    x evaluation point (scalar)
    n degree 
    Note d/dx T(x,n)= n*U(x,n-1)  
    """
    if n == 0:
        return 1.0
    if n == 1:
        return 2. * x
    return 2. * x * U(x, n - 1) - U(x, n - 2) 
    
def chebyquad_fcn(x):
    """
    Nonlinear function: R^n -> R^n
    """    
    n = len(x)
    def exact_integral(n):
        """
        Generator object to compute the exact integral of
        the transformed Chebychev function T(2x-1,i), i=0...n
        """
        for i in range(n):
            if i % 2 == 0: 
                yield -1./(i**2 - 1.)
            else:
                yield 0.

    exint = exact_integral(n)
    
    def approx_integral(i):
        """
        Approximates the integral by taking the mean value
        of n sample points
        """
        return sum(T(2. * xj - 1., i) for xj in x) / n
    return array([approx_integral(i) - e for i,e in enumerate(exint)]) 

def chebyquad(x):
    """            
    norm(chebyquad_fcn)**2                
    """
    chq = chebyquad_fcn(x)
    return dot(chq, chq)

def gradchebyquad(x):
    """
    Evaluation of the gradient function of chebyquad
    """
    chq = chebyquad_fcn(x)
    UM = 4. / len(x) * array([[(i+1) * U(2. * xj - 1., i) 
                             for xj in x] for i in range(len(x) - 1)])
    return dot(chq[1:].reshape((1, -1)), UM).reshape((-1, ))
    
if __name__ == '__main__':
    # Task 11 - Minimizing chebyquad and compare the results to the results obtained from scipy.optimize.fmin_bfgs
    # Works for n = 4, for n>4 elements are the same but permuted? Converge to minimum, but maybe a different one.
    x=linspace(0,1,8)
    xmin= so.fmin_bfgs(chebyquad,x,gradchebyquad, disp = False)  # should converge after 18 iterations
    opt = OptimizationProblem(chebyquad, gradchebyquad)
    x_opt = opt.newton_solve(x,"bfgs")
    print(chebyquad(x_opt))
    print(chebyquad(xmin))
    print('x_opt:', x_opt, '\nx_teory:', xmin)




