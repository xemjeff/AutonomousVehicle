# Can self-calibrate for a center position
# Returns distance from center position in feet
# Finds change in gps coords. from last, in feet

from gps3 import gps3
from math import radians, cos, sin, asin, sqrt
import os,threading
from time import sleep

# Returns distance between two locations in feet
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956
    return (c * r) * 5280
   
gps_socket = gps3.GPSDSocket()
data_stream = gps3.DataStream()
gps_socket.connect()
gps_socket.watch()

class gps_monitor():
	def __init__(self):
		self.center_lon,self.center_lat,self.previous_lon,self.previous_lat,self.current_lon,self.current_lat,self.distance_traveled,self.distance_from_center = 0,0,0,0,0,0,0,0
		threading.Thread(target=self._get_data).start()

	def calibrate_center(self,lat,lon):
		self.center_lon = lon
		self.center_lat = lat

	def _get_data(self):
		while True:
			for new_data in gps_socket:
				if new_data:
					data_stream.unpack(new_data)
					if self.current_lat == 0: self.calibrate_center(data_stream.TPV['lat'],data_stream.TPV['lon'])
					self.current_lat = data_stream.TPV['lat']
					self.current_lon = data_stream.TPV['lon']

					if self.previous_lon != 0:
						self.distance_last_traveled=haversine(self.previous_lon,self.previous_lat,self.current_lon,self.current_lat)
						self.previous_lon = self.current_lon
						self.previous_lat = self.current_lat
					if self.center_lon != 0:
						self.distance_from_center=haversine(self.center_lon,self.center_lat,self.current_lon,self.current_lat)
			sleep(1)



