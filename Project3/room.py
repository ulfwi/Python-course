

class Room:

    def __init__(self, room_nbr, u):
        self.room_nbr = room_nbr
        self.u = u
        if room_nbr == 1:
            self.matrix_a = self.matrix_a1(u)
        elif room_nbr == 2:
            self.matrix_a = self.matrix_a2(u)
        else:
            self.matrix_a = self.matrix_a3(u)
        self.b = self.get_b()

    # Matrices

    def matrix_a1(self, u):
        """
        Matrix describing room 1, having Neumann conditions on GAMMA_1
        :param u: is u1
        :return:
        """
        pass
        """
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
        """
    def matrix_a2(self, u2):
        """
        Matrix describing temperatures in room 2, having Dirichlet conditions on GAMMA_1 and GAMMA_2
        :param u2:
        :return:
        """
        pass

    def matrix_a3(self, u3):
        """
        Matrix describing room 3, having Neumann conditions on GAMMA_2
        :param u3:
        :return:
        """
        pass

    def get_b(self):
        # return different depending on room_nbr
        pass