
"""
Project 2 FMNN25 Advanced Numerical Algorithms with Python/SciPy
Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)

"""""

import unittest
from optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class TestOptimizer(unittest.TestCase):


    def setUp(self):
        """
        Creates an optimizer object with objective function func

        :return: --
        """""
        # Rosenbrock function
        def func(x):
            return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
        self.opt = Optimizer(func)

    def tearDown(self):
        pass


    # Task 7:
    # Test this seperately from an optimization method on Rosenbrock’s function and use the parameters given on p.37 in
    # the book mentioned above.