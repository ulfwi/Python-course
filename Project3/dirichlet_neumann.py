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
        n = 1./self.dx # length of row/col for room 1 and 3, as well as row for room 2 (col=2*n)

        # create A2 matrix
        row = np.zeros(2*n**2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A2 = la.toeplitz(row,row)

        # find boundary node indices
        index_heater = np.arange(n)
        index_wall = np.arange(n,n*(n-1)+1,n)
        index_wall.append(np.arange(n*(n+1)-1,2*n**2,n))
        index_window = np.arange(n*(2*n-1),2*n**2,1)
        index_gamma1 = np.arange(n**2,(2*n-2)*n+1,n)
        index_gamma2 = np.arange(2*n-1,(2*n-1)*n,n)

        # dirichlet boundary conditions on wall, windows and internal boundary nodes











        for i in range(maxit):
            pass



    def solve_domain(self):
        pass



