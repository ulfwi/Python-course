from mpi4py import MPI

# =================================================== Main-file =================================================== #
# kolla deltax = 1 !!!

rank = MPI.COMM_WORLD.Get_rank()

# ---------------------------------------------------- Task 1 ----------------------------------------------------- #
'''
if rank == 0:

    # Mesh size
    dx = 1/3

    # Create rooms
    r1 = RoomOne(10, dx, temp_wall=15, temp_heater=40)
    r2 = RoomTwo(20, dx, temp_wall=15, temp_heater=40, temp_window=5)
    r3 = RoomThree(25, dx, temp_wall=15, temp_heater=40)

    # Print all matrices
    print('Matrix for room 1\n', r1.a)
    print('\n')
    print('Matrix for room 2\n', r2.a)
    print('\n')
    print('Matrix for room 3\n', r3.a)
    print('\n')
'''
# -------------------------------------------------- Task 2 & 3 --------------------------------------------------- #

'''
# Mesh size
dx = 1. / 20

# Create rooms
r1 = RoomOne(10, dx)
r2 = RoomTwo(20, dx)
r3 = RoomThree(25, dx)

# Create apartment
flat = Apartment(r1, r2, r3)

# Solve heat equation problem
flat.dirichlet_neumann()

if rank == 0:
    # Plot apartment
    flat.plot_apartment()

'''


# ---------------------------------------------------- Task 4 ----------------------------------------------------- #


# Optional...
