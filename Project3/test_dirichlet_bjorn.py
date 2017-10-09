import scipy.linalg as la
import numpy as np
from room import Room, RoomOne, RoomTwo, RoomThree
from apartment import Apartment

# outside = np.zeros([5, 5])
# room1 = np.ones([5, 5])
# room2 = np.ones([10, 5])
# room3 = np.ones([5, 5])
# heatmap = np.append(outside, room1, axis=0)
# heatmap = np.append(heatmap, room2, axis=1)
# room3_temp = np.append(room3, outside, axis=0)
# print(room3_temp)
# heatmap = np.append(heatmap, room3_temp, axis=1)
#
# extent = [0, 3, 0, 2]
#
# plt.clf()
# plt.imshow(heatmap, extent=extent, origin='upper')
# plt.colorbar()
# plt.title('Apartment heatmap')
# plt.show()

print('hej')
# To create rooms
dx = 1. / 3
r1 = RoomOne(20, dx)
print('room1')
r2 = RoomTwo(20, dx)
print('room2')
r3 = RoomThree(20, dx)
print('room3')

flat = Apartment(r1, r2, r3)
print('apart')

flat.plot_apartment()