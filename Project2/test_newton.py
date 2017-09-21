from optimizer import Optimizer
import numpy as np


def func(x):
    return -x*x

solver = Optimizer(func)
x_opt = solver.newton_method(np.array([10,10,10]), "finiteDiff", "exact")
print(x_opt)