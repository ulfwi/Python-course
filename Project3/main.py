from apartment import Apartment
from room import Room, RoomOne, RoomTwo, RoomThree



# To create rooms
r1 = RoomOne(dx=0.05, temp_wall=15, temp_window=5, temp_heater=40, temp_inside=20)
r2 = RoomOne(dx=0.05, temp_wall=15, temp_window=5, temp_heater=40, temp_inside=20)
r3 = RoomOne(dx=0.05, temp_wall=15, temp_window=5, temp_heater=40, temp_inside=20)

flat = Apartment(r1, r2, r3)


dn.dx