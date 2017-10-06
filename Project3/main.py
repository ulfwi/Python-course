from apartment import Apartment
from room import Room, RoomOne, RoomTwo, RoomThree


# To create rooms
dx = 1. / 20
r1 = RoomOne(20, dx, temp_wall=15, temp_heater=40, temp_window=5)
r2 = RoomTwo(20, dx, temp_wall=15, temp_heater=40, temp_window=5)
r3 = RoomThree(20, dx, temp_wall=15, temp_heater=40)

flat = Apartment(r1, r2, r3)
