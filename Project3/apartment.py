import scipy.linalg as la
import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Apartment:

    def __init__(self, r1, r2, r3):
        """
        Constructs an apartment consisting of three rooms of type RoomOne, RoomTwo and RoomThree

        :param r1: room of type RoomOne
        :param r2: room of type RoomTwo
        :param r3: room of type RoomThree
        """
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.comm = MPI.COMM_WORLD

        # Make sure two kernels (?!) are used
        if self.comm.Get_size() != 2:
            raise ValueError("The rank-size must be two")

        if not r1.n == r2.n == r3.n:
            raise ValueError("The rooms must have the same discretization dx")

    def dirichlet_neumann(self, maxit=10, omega=0.8):
        """
        Iterates maxit times solving the heat equation using the method dirichlet-neumann.

        :param maxit: number of iterations
        :param omega: relaxation parameter
        :return: --
        """

        rank = self.comm.Get_rank()

        for i in range(maxit):

            # Save old u:s
            u1_old = np.copy(self.r1.u)
            u2_old = np.copy(self.r2.u)
            u3_old = np.copy(self.r3.u)

            # Update room 2
            self.r2.update_b([self.r1.u[self.r1.index_gamma], self.r3.u[self.r3.index_gamma]])
            self.r2.u = la.solve(self.r2.a, self.r2.b)

            # Update room 1 and 3
            if rank == 0:
                self.r1.update_b(self.r2.u[self.r2.index_gamma[0] + 1])
                self.r1.u = la.solve(self.r1.a, self.r1.b)

                # Receive the data from process 1
                st = MPI.Status()
                self.r3 = self.comm.recv(source=1, tag=0, status=st)

            if rank == 1:
                self.r3.update_b(self.r2.u[self.r2.index_gamma[1] - 1])
                self.r3.u = la.solve(self.r3.a, self.r3.b)

                # Send the data to process 0
                self.comm.send(self.r3, dest=0, tag=0)

            # Relaxation
            self.r1.u = omega * self.r1.u + (1 - omega) * u1_old
            self.r2.u = omega * self.r2.u + (1 - omega) * u2_old
            self.r3.u = omega * self.r3.u + (1 - omega) * u3_old

    def plot_apartment(self):
        """
        Plots a heatmap of the apartment. Areas outside apartment has
        temperature set to zero degrees to distinguish it from the apartment.

        :return: --
        """
        rank = self.comm.Get_rank()
        if rank == 0:
            n = self.r1.n
            outside = np.ones([n, n])*(self.r2.temp_window - 5)  # assume 5 degrees colder outside than window_temp
            u1 = np.copy(self.r1.u)
            u1.resize(n, n)
            u2 = np.copy(self.r2.u)
            u2.resize(2*n, n)
            u3 = np.copy(self.r3.u)
            u3.resize(n, n)

            # Combining all the rooms temperatures to a matrix
            heatmap = np.append(outside, u1, axis=0)
            heatmap = np.append(heatmap, u2, axis=1)
            room3_temp = np.append(u3, outside, axis=0)
            heatmap = np.append(heatmap, room3_temp, axis=1)

            extent = [0, 3, 0, 2]

            plt.figure()
            plt.imshow(heatmap, extent=extent, origin='upper')
            plt.colorbar()
            plt.title('Apartment heatmap')
            plt.show()

