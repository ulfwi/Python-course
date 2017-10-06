import scipy.linalg as la
import numpy as np
from mpi4py import MPI
from room import Room, RoomOne, RoomTwo, RoomThree
import matplotlib.pyplot as plt


class Apartment:

    def __init__(self, r1, r2, r3, omega=0.8):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.omega=omega

        if MPI.COMM_WORLD.Get_size() != 2:
            raise ValueError("The rank-size must be two")

    def dirichlet_neumann(self, u1, u2, u3, maxit=10):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            N = 1/self.dx  # number of square elements.

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

                # Receive the data from process 1
                st = MPI.Status()
                r3 = comm.recv(source=1, tag=0, status=st)

            if rank == 1:
                r3.get_b(r2.u(r2.gamma))
                r3.u = la.solve(r3.a, r3.b)

                # Send the data to process 0
                comm.send(r3, dest=0, tag=0)

            # Relaxation
            r1.u = self.omega * r1.u + (1 - self.omega) * u1_old
            r2.u = self.omega * r2.u + (1 - self.omega) * u2_old
            r3.u = self.omega * r3.u + (1 - self.omega) * u3_old

    def solve_domain(self):
        pass

    def plot_apartment(self):
        """
        Plots a heatmap of the apartment. Areas outside apartment has
        temperature set to zero degrees to distinguish it from the apartment.
        :return: ~~
        """
        n = self.r1.n
        outside = np.zeros([n, n])
        u1 = self.r1.u
        u2 = self.r2.u
        u3 = self.r3.u
        # combining all the rooms temperatures to a matrix
        heatmap = np.append(outside, u1, axis=0)
        heatmap = np.append(heatmap, u2, axis=1)
        room3_temp = np.append(u3, outside, axis=0)
        heatmap = np.append(heatmap, room3_temp, axis=1)

        extent = [0, 3, 0, 2]

        plt.clf()
        plt.imshow(heatmap, extent=extent, origin='upper')
        plt.colorbar()
        plt.title('Apartment heatmap')
        plt.show()










