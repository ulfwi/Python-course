from optimizer import Optimizer
import numpy as np


lb = True
fl = False

# Bjorn Lina
if lb:
    def func(x):
        return -np.dot(x,x)

    solver = Optimizer(func)
    x_opt = solver.newton_method(np.array([1.,1.,1.]), "dfp", "exact")
    print(x_opt)


# Fanny Louise
if fl:
    def f(x):
        return np.dot(x,x)
        # Rosenbrock function
        #return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

    p = np.array([2,2])
    x = np.array([-1,-1])
    Opt = Optimizer(f)

    alpha = Opt.ls_exact(p,x)
    print('alpha:', alpha)



