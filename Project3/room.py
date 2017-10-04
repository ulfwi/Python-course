from abc import ABC, abstractmethod


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
        pass


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
        pass


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
        pass



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

