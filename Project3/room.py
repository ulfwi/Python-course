from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as la
#import matplotlib.pyplot as plt


class Room(ABC):

    def __init__(self, dx, temp_wall=15, temp_heater=40):
        """
        Abstract room constructor with general class attributes.

        :param temp_wall: wall temperature
        :param temp_heater: heater temperature
        """
        if dx > 1:
            raise ValueError("The grid width must be smaller than 1 (the length of the room)")
        self.a = self.get_a()
        self.temp_wall = temp_wall
        self.temp_heater = temp_heater
        self.b = self.get_b()

    @abstractmethod
    def get_b(self):
        pass

    @abstractmethod
    def update_b(self, u):
        pass

    @abstractmethod
    def get_a(self):
        pass
    #
    # @abstractmethod
    # def plot_temp(self):
    #     pass


class RoomOne(Room):

    def __init__(self, temp_init=20, dx=1./20., temp_wall=15, temp_heater=40):
        """
        Construct a room of type one.

        :param temp_init: initial temperature distribution (constant)
        :param dx: grid width
        :param temp_wall: wall temperature
        :param temp_heater: heater temperature
        """
        self.n = int(1./dx) + 1
        self.index_heater, self.index_wall, self.index_gamma = self.get_boundaries()
        Room.__init__(self, dx, temp_wall, temp_heater)
        self.u = self.init_u(temp_init)

    def get_boundaries(self):
        """
        Find boundary indices corresponding to wall, heater and internal boundary gamma.

        :return: index_heater, index_wall, index_gamma
        """
        n = self.n
        index_heater = np.arange(n, (n - 2) * n + 1, n)
        index_wall = np.append(np.arange(n), np.arange(n * (n - 1), n ** 2))
        index_gamma = np.arange(2 * n - 1, (n - 1) * n, n)

        return index_heater, index_wall, index_gamma

    def init_u(self, temp_init):
        """
        Initialize constant temperature distribution u using temperature temp_init.

        :param temp_init: initial room temperature
        :return: u
        """
        n = self.n
        u = np.ones(n ** 2) * temp_init
        u[self.index_heater] = self.temp_heater
        u[self.index_wall] = self.temp_wall
        return u

    def get_b(self):
        """
        Calculates right hand side b using fixed Dirichlet boundary conditions.

        :return: b
        """
        n = self.n
        b = np.zeros(n ** 2)

        b[self.index_wall] = self.temp_wall
        b[self.index_heater] = self.temp_heater

        return b

    def update_b(self, u2gamma):
        """
        Updates b on internal boundary using Neumann conditions.

        :param u2gamma: temperatures u(gamma1 + 1) from room 2
        :return: --
        """
        grad = u2gamma - self.u[self.index_gamma]
        self.b[self.index_gamma] = -grad

    def get_a(self):
        """
        Matrix describing discretized heat eq in room one, having Neumann conditions on boundary gamma 1.

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
        indices = np.array(list(self.index_heater) + list(self.index_wall))

        # dirichlet boundary conditions on wall and heater
        A[indices] = np.zeros(n ** 2)
        A[indices, indices] = 1.

        # neumann boundary conditions on internal boundary
        A[self.index_gamma, self.index_gamma] = -3.
        A[self.index_gamma, self.index_gamma + 1] = 0.

        return A
    #
    # def plot_temp(self):
    #     """
    #     Plots the temperature distribution of the room.
    #
    #     :return: --
    #     """
    #     uplot = np.copy(self.u)
    #     uplot.resize(self.n, self.n)
    #     extent = [0, 1, 0, 1] # xmin xmax ymin ymax
    #     plt.clf()
    #     plt.imshow(uplot, extent=extent, origin='upper')
    #     plt.title('Temperature distribution in Room one')
    #     plt.show()


class RoomTwo(Room):

    def __init__(self, temp_init=20, dx=1./20., temp_wall=15, temp_heater=40, temp_window=5):
        """
        Construct a room of type two.

        :param temp_init: initial temperature distribution
        :param dx: grid width
        :param temp_wall: wall temperature
        :param temp_heater: heater temperature
        :param temp_window: window temperature
        """
        self.n = int(1./dx) + 1
        self.index_heater, self.index_wall, self.index_gamma, self.index_window = self.get_boundaries()
        self.temp_window = temp_window
        Room.__init__(self, dx, temp_wall, temp_heater)
        self.u = self.init_u(temp_init)

    def get_boundaries(self):
        """
        Find boundary indices corresponding to wall, window, heater and internal boundary gamma.

        :return: index_heater, index_wall, index_gamma, index_window
        """
        n = self.n
        index_heater = np.arange(1, n - 1)
        index_wall = np.concatenate((np.arange(0, n ** 2 + 1, n), np.arange((n ** 2) - 1, 2 * n ** 2, n),
                                     np.array([n - 1, n * (2 * n - 1)])))
        index_window = np.arange(n * (2 * n - 1) + 1, 2 * n ** 2 - 1)
        index_gamma1 = np.arange(n * (n + 1), (2 * n - 2) * n + 1, n)
        index_gamma2 = np.arange(2 * n - 1, (n - 1) * n, n)
        index_gamma = [index_gamma1, index_gamma2]

        return index_heater, index_wall, index_gamma, index_window

    def init_u(self, temp_init):
        """
        Initialize constant temperature distribution u using temperature temp_init.

        :param temp_init: initial room temperature
        :return: u
        """
        n = self.n
        u = np.ones(2 * self.n ** 2)*temp_init

        u[self.index_heater] = self.temp_heater
        u[self.index_wall] = self.temp_wall
        u[self.index_window] = self.temp_window
        return u

    def get_b(self):
        """
        Construct b array using constant Dirichlet conditions on external boundaries.

        :return: b
        """
        n = self.n
        b = np.zeros(2 * n ** 2)
        b[self.index_wall] = self.temp_wall
        b[self.index_heater] = self.temp_heater
        b[self.index_window] = self.temp_window
        return b

    def update_b(self, ugamma):
        """
        Update b on internal boundaries gamma 1 and gamma 2 using Dirichlet conditions.

        :param ugamma: known temperatures on internal boundaries from adjacent rooms
        """
        self.b[self.index_gamma[0]] = ugamma[0]
        self.b[self.index_gamma[1]] = ugamma[1]

    def get_a(self):
        """
        Matrix describing discretized heat eq in room two, having Dirichlet conditions on GAMMA_1 and GAMMA_2.

        :return: matrix A
        """
        n = self.n
        # create A matrix
        row = np.zeros(2 * n ** 2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A = la.toeplitz(row, row)

        # find boundary node indices

        indices = np.array(list(self.index_heater) + list(self.index_wall) + list(self.index_window)
                           + list(self.index_gamma[0]) + list(self.index_gamma[1]))

        # dirichlet boundary conditions on wall, heater, windows and internal boundary nodes
        A[indices] = np.zeros(2 * n ** 2)
        A[indices, indices] = 1

        return A

    # def plot_temp(self):
    #     """
    #     Plots the temperature distribution of the room.
    #
    #     :return: --
    #     """
    #     uplot = np.copy(self.u)
    #     uplot.resize(2 * self.n, self.n)
    #     extent = [0, 1, 0, 1]  # xmin xmax ymin ymax
    #     plt.clf()
    #     plt.imshow(uplot, extent=extent, origin='upper')
    #     plt.title('Temperature distribution in Room two')
    #     plt.show()


class RoomThree(Room):
    def __init__(self, temp_init=20, dx=1./20., temp_wall=15, temp_heater=40):
        """
        Construct a room of type three.

        :param temp_init: initial temperature distribution
        :param dx: grid width
        :param temp_wall: wall temperature
        :param temp_heater: heater temperature
        """
        self.n = int(1./dx) + 1
        self.index_heater, self.index_wall, self.index_gamma = self.get_boundaries()
        Room.__init__(self, dx, temp_wall, temp_heater)
        self.u = self.init_u(temp_init)

    def get_boundaries(self):
        """
        Find boundary indices corresponding to wall, heater and internal boundary gamma.

        :return: index_heater, index_wall, index_gamma
        """
        n = self.n
        index_wall = np.append(np.arange(n), np.arange(n * (n - 1), n ** 2))
        index_heater = np.arange(2 * n - 1, (n - 1) * n, n)
        index_gamma = np.arange(n, n * (n - 2) + 1, n)

        return index_heater, index_wall, index_gamma

    def init_u(self, temp_init):
        """
        Initialize constant temperature distribution u using temp_init.

        :param temp_init: initial room temperature
        :return: u
        """
        n = self.n
        u = np.ones(n ** 2) * temp_init
        u[self.index_heater] = self.temp_heater
        u[self.index_wall] = self.temp_wall
        return u

    def get_b(self):
        """
        Construct b array using constant Dirichlet conditions on external boundaries.

        :return: b
        """
        n = self.n
        b = np.zeros(n ** 2)
        # Find boundary node indices
        b[self.index_wall] = self.temp_wall
        b[self.index_heater] = self.temp_heater
        # let gamma nodes be unknown in constructor
        return b

    def update_b(self, u2gamma):
        """
        Update b on internal boundary using Neumann conditions.

        :param u2gamma: temperatures u(gamma2 - 1) from room 2
        :return: --
        """
        grad =  u2gamma - self.u[self.index_gamma]
        self.b[self.index_gamma] = -grad

    def get_a(self):
        """
        Matrix describing discretized heat eq in room three, having Neumann conditions on GAMMA_2.

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
        indices = np.array(list(self.index_heater) + list(self.index_wall))

        # dirichlet boundary conditions on wall and heater
        A[indices] = np.zeros(n ** 2)
        A[indices, indices] = 1

        # neumann boundary conditions on internal boundary
        A[self.index_gamma, self.index_gamma] = -3
        A[self.index_gamma, self.index_gamma - 1] = 0

        return A

    # def plot_temp(self):
    #     """
    #     Plots the temperature distribution in the room.
    #
    #     :return: --
    #     """
    #     uplot = np.copy(self.u)
    #     uplot.resize(self.n, self.n)
    #     extent = [0, 1, 0, 1]  # xmin xmax ymin ymax
    #     plt.clf()
    #     plt.imshow(uplot, extent=extent, origin='upper')
    #     plt.title('Temperature distribution in Room three')
    #     plt.show()


