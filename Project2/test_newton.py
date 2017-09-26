from optimizer import OptimizationProblem
import numpy as np


lb = False
fl = True


# Bjorn Lina
if lb:
    def func_rosenbrock(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def grad_rosenbrock(x):
        return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])

    def func(x):
        #return np.dot(x,x)
        return x[0]*x[0] + 0.5*(x[1]-5)*(x[1]-5)
        #return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    solver = OptimizationProblem(func_rosenbrock, grad_rosenbrock)
    x_opt = solver.newton_method(np.array([0.,0.]), "finiteDiff", "exact", 10**-6, 1000)
    print(x_opt)


# Fanny Louise
if fl:
    def f(x):
        return np.dot(x,x)
        # Rosenbrock function
        #return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

    p = np.array([2,2])
    x = np.array([-1,-1])
    Opt = OptimizationProblem(f)

    #alpha = Opt.ls_exact(p,x)
    #print('alpha:', alpha)



