import scipy.linalg as la
import numpy as np
from mpi4py import MPI
import scipy as sp


class DirichletNeumann:

    def __init__(self, dx=0.05, temp_wall=15, temp_window=5, temp_heater=40):
        self.dx = dx
        self.temp_wall = temp_wall

    def dirichlet_neumann(self, u01, u02, u03, omega=0.8, maxit=10):
        u1 = np.copy(u01)
        u2 = np.copy(u02)
        u3 = np.copy(u03)
        n1 = 1./self.dx # length of row/col
        n2row = 1./self.dx # length of row
        n2col = 2./self.dx # length of column
        n3 = 1./self.dx # length of row/col
        #A1 = np.zeros([n1,n1])
        # Write separate method to generate A
        Bdiag = -4*np.eye(n1)
        Bupper = np.diag([1] * (n1 - 1), 1)
        Blower = np.diag([1] * (n1 - 1), -1)
        B1 = Bdiag + Bupper + Blower
        blist = [B1] * n1 # list of n1 Bs
        A1 = sp.linalg.block_diag(*blist)
        Dlower = np.diag(np.ones(n1*(n1-1)), n1)
        Dupper = np.diag(np.ones(n1*(n1-1)), -n1)
        A1 += Dupper + Dlower

        A2 = np.zeros([n2row*n2col,n2row*n2col])
        A3 = np.zeros([n3**2,n3**2])


        for i in range(maxit):
            pass



    def solve_domain(self):
        pass



