import scipy.linalg as la
import numpy as np
from mpi4py import MPI
from room import Room, RoomOne, RoomTwo, RoomThree


class DirichletNeumann:

    def __init__(self, dx=0.05, temp_wall=15, temp_window=5, temp_heater=40, omega=0.8):
        self.dx = dx
        self.temp_wall = temp_wall
        self.omega=omega

        if MPI.COMM_WORLD.Get_size() != 2:
            raise ValueError("The rank-size must be two")

    def dirichlet_neumann(self, u1, u2, u3, maxit=10):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
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
            if rank == 0:
                r1.get_b(r2.u(r2.gamma))              # update_b?
                r1.u = la.solve(r1.a, r1.b)

                #Recieve the data from process 1
                st = MPI.Status()
                r3 = comm.recv(source=1, tag=0, status=st)

            if rank == 1:
                r3.get_b(r2.u(r2.gamma))
                r3.u = la.solve(r3.a, r3.b)

                #Send the data to process 0
                comm.send(r3, dest=0, tag=0)

            # Relaxation
            r1.u = self.omega * r1.u + (1 - self.omega) * u1_old
            r2.u = self.omega * r2.u + (1 - self.omega) * u2_old
            r3.u = self.omega * r3.u + (1 - self.omega) * u3_old


    def solve_domain(self):
        pass










