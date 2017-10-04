import scipy.linalg as la
import numpy as np
from mpi4py import MPI
from room import Room, RoomOne, RoomTwo, RoomThree


class DirichletNeumann:

    def __init__(self, dx=0.05, temp_wall=15, temp_window=5, temp_heater=40):
        self.dx = dx
        self.temp_wall = temp_wall

    def dirichlet_neumann(self, u1, u2, u3, omega=0.8, maxit=10):

        N = 1/self.dx # number of square elements.              of row/col for room 1 and 3, as well as row for room 2 (col=2*n)

        # Create rooms
        r1 = RoomOne(u1)
        r2 = RoomTwo(u2)
        r3 = RoomThree(u3)

        for i in range(maxit):
            # Save old u:s
            u1_old = np.copy(r1.u)
            u2_old = np.copy(r2.u)
            u3_old = np.copy(r3.u)

            # Update room 2
            r2.u = la.solve(r2.a, r2.b)

            # Update room 1 and 3
            r1.get_b()              # update_b?
            r3.get_b()
            r1.u = la.solve(r1.a, r1.b)
            r3.u = la.solve(r3.a, r3.b)

            # Relaxation




    def solve_domain(self):
        pass






