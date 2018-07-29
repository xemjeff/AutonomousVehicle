# Can self-calibrate for a center position
# Returns distance from center position in feet
# Finds change in gps coords. from last, in feet

from gps3 import gps3
from math import radians, cos, sin, asin, sqrt
import redis, os

# Returns distance between two locations in feet
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956
    return (c * r) * 5280
    
memory = redis.StrictRedis(host='localhost',port=6379,db=0)

os.system('gpsd /dev/ttyAMA0')

gps_socket = gps3.GPSDSocket()
data_stream = gps3.DataStream()
gps_socket.connect()
gps_socket.watch()

# Input the lon,lat coords. here that you want the robot to stay 30ft. within
center_lon, center_lat = 0,0

previous_lon, previous_lat = 0,0

for new_data in gps_socket:
	if new_data:
		data_stream.unpack(new_data)
		lat = data_stream.TPV['lat']
		lon = data_stream.TPV['lon']
                if memory.get('calibrate_center_gps') == 'True':
                    memory.set('calibrate_center_gps','False')
                    center_lat = lat
                    center_lon = lon
                else:
                    memory.set('gps_latitude',lat)
                    memory.set('gps_longitude',lon)
                    if previous_lon != 0:
                        memory.set('distance_last_traveled',haversine(previous_lon,previous_lat,lon,lat))
                        previous_lon = lon
                        previous_lat = lat
                    if center_lon != 0:
                        memory.set('distance_from_center',haversine(center_lon,center_lat,lon,lat))
