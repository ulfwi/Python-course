from mpi4py import MPI
from room import Room, RoomOne, RoomTwo, RoomThree
from apartment import Apartment
import numpy as np


class RunMain:

    def __init__(self):
        pass

# =================================================== Main-file =================================================== #

    # Defining mpi
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()

    # User gets to chose task
    task = None
    if rank == 0:
        input_not_ok = True
        while input_not_ok:
            task = input('Choose a task: ')
            if task == '1' or task == '2' or task == '3' or task == '4':
                task = int(task)
                print('You chose task : ' + str(task) + '\n')
                input_not_ok = False
            else:
                print('You must chose task 1, 2, 3' or 4)

    task = comm.bcast(task, root=0)


# ---------------------------------------------------- Task 1 ----------------------------------------------------- #

    if task == 1:
        if rank == 0:

            # Mesh size
            dx = 1/3

            # Create rooms
            r1 = RoomOne(10, dx)
            r2 = RoomTwo(20, dx)
            r3 = RoomThree(25, dx)

            # Print all matrices
            print('Matrix for room 1\n', r1.a)
            print('\n')
            print('Matrix for room 2\n', r2.a)
            print('\n')
            print('Matrix for room 3\n', r3.a)
            print('\n')

# -------------------------------------------------- Task 2 & 3 --------------------------------------------------- #

    if task == 2 or task == 3:
        # Mesh size
        dx = 1/20

        # Create rooms
        r1 = RoomOne(20, dx)
        r2 = RoomTwo(20, dx)
        r3 = RoomThree(20, dx)

        # Create apartment
        flat = Apartment(r1, r2, r3)

        # Solve heat equation problem
        flat.dirichlet_neumann()

        if rank == 0:
            # Calculate mean temperature
            u_mean = np.mean(np.concatenate((flat.r1.u, flat.r2.u, flat.r3.u)))
            print('Mean temperature: ' + str(u_mean))
            # Plot apartment
            flat.plot_apartment()

# ---------------------------------------------------- Task 4 ----------------------------------------------------- #


    # Optional assignment

    # Try other initial temperatures

    if task == 4:

        # Mesh size
        dx = 1/30
        temp_wall=15
        temp_window=-10

        # Create rooms
        r1 = RoomOne(temp_init=10, dx, temp_wall, temp_heater=30)
        r2 = RoomTwo(temp_init=20, dx, temp_wall, temp_heater=40, temp_window)
        r3 = RoomThree(temp_init=25, dx, temp_wall, temp_heater=50)


        # Create apartment
        flat = Apartment(r1, r2, r3)

        # Solve heat equation problem
        flat.dirichlet_neumann()

        if rank == 0:
            # Calculate mean temperature
            u_mean = np.mean(np.concatenate((flat.r1.u, flat.r2.u, flat.r3.u)))
            print('Mean temperature: ' + str(u_mean))

            # Plot apartment
            flat.plot_apartment()


if __name__ == '__main__':
    main()