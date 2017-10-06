from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as la
#import plotly as py
#from plotly.graph_objs import *
import matplotlib.pyplot as plt



class Room(ABC):

    def __init__(self, dx, temp_wall=15, temp_heater=40):
        self.n = int(1./dx)
        self.gamma = self.get_gamma()
        self.a = self.get_a()
        self.temp_wall = temp_wall
        self.temp_heater = temp_heater
        self.b = self.get_b()

    @abstractmethod
    def get_gamma(self):
        pass

    @abstractmethod
    def get_b(self):
        pass

    @abstractmethod
    def update_b(self, u):
        raise NotImplementedError('subclasses must override Room()!')

    @abstractmethod
    def get_a(self):
        pass

    @abstractmethod
    def plot_temp(self):
        pass


class RoomOne(Room):

    def __init__(self, temp_init, dx, temp_wall=15, temp_heater=40, temp_window=5):
        Room.__init__(self, dx, temp_wall, temp_heater)
        self.u = np.ones(self.n ** 2) * temp_init
        self.temp_window = temp_window

    def get_gamma(self):
        """
        Calculates indices of internal boundary nodes.

        :return: indices of internal boundary nodes
        """
        n = self.n
        return np.arange(2 * n - 1, (n - 1) * n, n)

    def get_b(self):
        """
        Calculates right hand side b using fixed Dirichlet boundary conditions.

        :return: array b
        """
        n = self.n
        b = np.zeros(n ** 2)

        index_heater = np.arange(n, (n - 2) ** 2 + 1, n)
        index_wall = np.append(np.arange(n), np.arange(n * (n - 1), n ** 2))

        b[index_wall] = self.temp_wall
        b[-1] = self.temp_window # fix temp = 5 at conflict node
        b[index_heater] = self.temp_heater

        return b

    def update_b(self, u2gamma):
        """
        Updates b on internal boundary using Neumann conditions.

        :param u2gamma: temperatures u(gamma1 + 1) from room 2
        :return: --
        """
        grad = u2gamma - self.u[self.gamma]
        self.b[self.gamma] = -grad

    def get_a(self):
        """
        Matrix describing room 1, having Neumann conditions on GAMMA_1.

        :param u: is u1
        :return: matrix A
        """
        n = self.n

        # create A matrix
        row = np.zeros(n ** 2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A = la.toeplitz(row, row)

        # find boundary node indices
        index_heater = np.arange(n, (n - 2) ** 2 + 1, n)
        index_wall = np.append(np.arange(n), np.arange(n * (n - 1), n ** 2))
        indices = np.array(list(index_heater) + list(index_wall))

        # dirichlet boundary conditions on wall and heater
        A[indices] = np.zeros(n ** 2)
        A[indices, indices] = 1.

        # neumann boundary conditions on internal boundary
        A[self.gamma, self.gamma] = -3.
        A[self.gamma, self.gamma + 1] = 0.

        return A

    def plot_temp(self, filename='Room_1_plot.html'):
        uplot = np.copy(self.u)
        uplot.resize(self.n, self.n)
        extent = [0, 1, 0, 1] # xmin xmax ymin ymax
        plt.clf()
        plt.imshow(uplot, extent=extent, origin='upper')
        plt.title('Temperature distribution in Room one')
        plt.show()




class RoomTwo(Room):

    def __init__(self, temp_init, dx, temp_wall=15, temp_heater=40, temp_window=5):
        self.temp_window = temp_window
        Room.__init__(self, dx, temp_wall, temp_heater)
        self.u = np.ones(2 * self.n ** 2)*temp_init
        self.update_b(self.u[self.gamma])

    def get_gamma(self):
        """
        Calculates indices of internal boundary nodes.

        :return: indices of internal boundary nodes
        """
        n = self.n
        index_gamma1 = np.arange(n ** 2, (2 * n - 2) * n + 1, n)
        index_gamma2 = np.arange(2 * n - 1, (n - 1) * n, n)
        return np.append(index_gamma1, index_gamma2)

    def get_b(self):
        """
        Construct b array using constant Dirichlet conditions on external boundaries.

        :return: b: Right hand side
        """
        n = self.n
        index_heater = np.arange(1, n - 1)
        index_wall = np.append(np.arange(0, n * (n - 1) + 1, n), np.arange((n ** 2) - 1, (2 * n - 1) * n, n))
        index_window = np.arange(n * (2 * n - 1), 2 * n ** 2)
        b = np.zeros(2 * n ** 2)
        b[index_wall] = self.temp_wall
        b[index_heater] = self.temp_heater
        b[index_window] = self.temp_window
        return b

    def update_b(self, ugamma):
        """
        Update b on internal boundaries gamma 1 and gamma 2 using Dirichlet conditions.

        :param ugamma: known temperatures on internal boundaries from adjacent rooms
        """
        self.b[self.gamma] = ugamma

    def get_a(self):
        """
        Matrix describing temperatures in room 2, having Dirichlet conditions on GAMMA_1 and GAMMA_2
        :param u: is u2
        :return:
        """
        n = self.n
        # create A matrix
        row = np.zeros(2 * n ** 2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A = la.toeplitz(row, row)

        # find boundary node indices
        index_heater = np.arange(1, n - 1)
        index_wall = np.append(np.arange(0, n * (n - 1) + 1, n), np.arange((n ** 2) - 1, (2 * n - 1) * n, n))
        index_window = np.arange(n * (2 * n - 1), 2 * n ** 2)
        indices = np.array(list(index_heater) + list(index_wall) + list(index_window) +
                           list(self.gamma))

        # dirichlet boundary conditions on wall, heater, windows and internal boundary nodes
        A[indices] = np.zeros(2 * n ** 2)
        A[indices, indices] = 1

        return A

    def plot_temp(self):
        uplot = np.copy(self.u)
        uplot.resize(2 * self.n, self.n)
        extent = [0, 1, 0, 1]  # xmin xmax ymin ymax
        plt.clf()
        plt.imshow(uplot, extent=extent, origin='upper')
        plt.title('Temperature distribution in Room two')
        plt.show()


class RoomThree(Room):

    def __init__(self, temp_init, dx, temp_wall=15, temp_heater=40):
        # self.n = int(np.sqrt(len(u)))
        Room.__init__(self, temp_init, dx, temp_wall, temp_heater)
        self.u = np.ones(self.n ** 2) * temp_init


    def get_gamma(self):
        """
        Calculates indices of internal boundary nodes.

        :return: indices of internal boundary nodes
        """
        n = self.n
        return np.arange(n, (n - 2) ** 2 + 1, n)

    def get_b(self):
        """
        Construct b array using constant Dirichlet conditions on external boundaries.

        :return: b: Right hand side
        """
        n = self.n
        b = np.zeros(n ** 2)
        # Find boundary node indices
        index_wall = np.append(np.arange(n), np.arange(n * (n - 1), n ** 2))
        index_heater = np.arange(2 * n - 1, (n - 1) * n, n)
        b[index_wall] = self.temp_wall
        b[index_heater] = self.temp_heater
        # let gamma nodes be unknown in constructor
        return b

    def update_b(self, u2gamma):
        """
        Updates b on internal boundary using Neumann conditions.

        :param u2gamma: temperatures u(gamma2 - 1) from room 2
        :return: --
        """
        grad = self.u[self.gamma] - u2gamma
        self.b[self.gamma] = -grad


    def get_a(self):
        """
        Matrix describing room 3, having Neumann conditions on GAMMA_2
        :param u: is u3
        :return:
        """
        n = self.n

        # create A matrix
        row = np.zeros(n ** 2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A = la.toeplitz(row, row)

        # find boundary node indices
        index_gamma = self.get_gamma()
        index_wall = np.append(np.arange(n), np.arange(n * (n - 1), n ** 2))
        index_heater = np.arange(2 * n - 1, (n - 1) * n, n)
        indices = np.array(list(index_heater) + list(index_wall))

        # dirichlet boundary conditions on wall and heater
        A[indices] = np.zeros(n ** 2)
        A[indices, indices] = 1

        # neumann boundary conditions on internal boundary
        A[index_gamma, index_gamma] = -3
        A[index_gamma, index_gamma - 1] = 0

        return A

    def plot_temp(self):
        uplot = np.copy(self.u)
        uplot.resize(self.n, self.n)
        extent = [0, 1, 0, 1]  # xmin xmax ymin ymax
        plt.clf()
        plt.imshow(uplot, extent=extent, origin='upper')
        plt.title('Temperature distribution in Room three')
        plt.show()


