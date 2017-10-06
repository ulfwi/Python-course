from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as la

class Room(ABC):

    @abstractmethod
    def get_gamma(self):
        pass

    @abstractmethod
    def get_b(self):
        raise NotImplementedError('subclasses must override foo()!')

    @abstractmethod
    def get_a(self, u):
        pass


class RoomOne(Room):

    def __init__(self, u):
        self.u = u
        self.gamma = self.get_gamma()
        self.a = self.get_a(u)
        self.b = None

    def get_gamma(self):
        pass

    def get_b(self):
        pass

    def get_a(self, u):
        """
        Matrix describing room 1, having Neumann conditions on GAMMA_1
        :param u: is u1
        :return:
        """
        n = np.sqrt(len(u))

        # create A matrix
        row = np.zeros(n ** 2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A = la.toeplitz(row, row)

        # find boundary node indices
        index_heater = np.arange(n, (n - 2) ** 2 + 1, n)
        index_wall = np.arange(n)
        index_wall.append(np.arange(n * (2 * n - 1), 2 * n ** 2))
        index_gamma = np.arange(2 * n - 1, (n - 1) * n, n)
        indices = np.array(list(index_heater) + list(index_wall))

        # dirichlet boundary conditions on wall and heater
        A[indices] = np.zeros(n)
        A[indices, indices] = 1

        # neumann boundary conditions on internal boundary
        A[index_gamma, index_gamma] = -3
        A[index_gamma, index_gamma + 1] = 0

        return A


class RoomTwo(Room):

    def __init__(self, u):
        self.u = u
        self.gamma = self.get_gamma()
        self.a = self.get_a(u)
        self.b = None

    def get_gamma(self):
        pass

    def get_b(self):
        pass

    def get_a(self, u):
        """
        Matrix describing temperatures in room 2, having Dirichlet conditions on GAMMA_1 and GAMMA_2
        :param u: is u2
        :return:
        """
        n = np.sqrt(len(u) / 2.)
        # create A matrix
        row = np.zeros(2 * n ** 2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A = la.toeplitz(row, row)

        # find boundary node indices
        index_heater = np.arange(1, n - 1)
        index_wall = np.arange(0, n * (n - 1) + 1, n)
        index_wall.append(np.arange((n ** 2) - 1, (2 * n - 1) * n, n))
        index_window = np.arange(n * (2 * n - 1), 2 * n ** 2)
        index_gamma1 = np.arange(n ** 2, (2 * n - 2) * n + 1, n)
        index_gamma2 = np.arange(2 * n - 1, (n - 1) * n, n)
        indices = np.array(list(index_heater) + list(index_wall) + list(index_window) +
                           list(index_gamma1) + list(index_gamma2))

        # dirichlet boundary conditions on wall, heater, windows and internal boundary nodes
        A[indices] = np.zeros(n)
        A[indices, indices] = 1

        return A


class RoomThree(Room):

    def __init__(self, u):
        self.u = u
        self.gamma = self.get_gamma()
        self.a = self.get_a(u)
        self.b = None

    def get_gamma(self):
        pass

    def get_b(self):
        pass

    def get_a(self, u):
        """
        Matrix describing room 3, having Neumann conditions on GAMMA_2
        :param u: is u3
        :return:
        """
        n = np.sqrt(len(u))

        # create A matrix
        row = np.zeros(n ** 2)
        row[0] = -4.
        row[1] = 1.
        row[n] = 1.
        A = la.toeplitz(row, row)

        # find boundary node indices
        index_gamma = np.arange(n, (n - 2) ** 2 + 1, n)
        index_wall = np.arange(n)
        index_wall.append(np.arange(n * (2 * n - 1), 2 * n ** 2))
        index_heater = np.arange(2 * n - 1, (n - 1) * n, n)
        indices = np.array(list(index_heater) + list(index_wall))

        # dirichlet boundary conditions on wall and heater
        A[indices] = np.zeros(n)
        A[indices, indices] = 1

        # neumann boundary conditions on internal boundary
        A[index_gamma, index_gamma] = -3
        A[index_gamma, index_gamma - 1] = 0

        return A



'''
def __init__(self, room_nbr, u):
    self.room_nbr = room_nbr
    self.u = u
    self.gamma = self.get_gamma()
    if room_nbr == 1:
        self.matrix_a = self.matrix_a1(u)
        self.b = None
    elif room_nbr == 2:
        self.matrix_a = self.matrix_a2(u)
        self.b = self.get_b()
    else:
        self.matrix_a = self.matrix_a3(u)
        self.b = None
'''

