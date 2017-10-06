
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

a = b = c = d = 0
if c == 0:
    raise ValueError("hmm.")


if rank == 0:
    a = 6.0
    b = 3.0
    print('a+b', a + b)
    st = MPI.Status()
    c = comm.recv(source=1, tag=2, status=st)
    print("%s (error=%d)" %(c, st.Get_error()))  # error = 0 is success
if rank == 1:
    c = 10
    print('a*b', a * b)
    comm.send(c, dest=0, tag=2)


'''
#if rank == 0:
time.sleep(2)
print('++', d + c + a)
t = np.array([a, b, c, d])
N = np.array([j for j in range(4)])
plt.plot(N, t, 'b')
plt.title('Testingtesting')
plt.show()
'''





'''
'''







'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    for i in range(1, size):
        sendMsg = "Hello, Rank %d" %i
        comm.send(sendMsg, dest=i)
else:
    recvMsg = comm.recv(source=0)
    print(recvMsg)
'''



'''
time.sleep(2)
a = 6
print(a + b)
print(a * b)
a = 10
print(max(a,b))
'''



'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#print(rank)
print("hello world from process ", rank)
'''


