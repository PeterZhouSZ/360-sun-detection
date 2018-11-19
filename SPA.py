import math
import csv
import os
import argparse
import numpy as np
import math
from misc.sunposition import sunpos, observed_sunpos
from datetime import datetime, timedelta
import time

class SPA_data(object):
	def __init__(self, lat, lng, heading, time):
		self.LAT = lat
		self.LNG = lng
		self.HEAD = heading
		self.TIME = time	
	def get_sunpos(self):
		#print(self.LAT, self.LNG)
		az,al = sunpos(self.TIME, self.LAT, self.LNG, 0)[:2]
		altitude = - (90 - al)
		azimuth = az - self.HEAD - 90
		if(azimuth < -270):
			azimuth = 90 - (-270 - azimuth)
		return azimuth, altitude
	

def readSPAData(fname):
	data  = []
	with open(fname, newline='') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			date = row['Date']
			time = row['Time']
			heading = float(row['Heading'])
			d_time = datetime(int(date.split('/')[0]), int(date.split('/')[1]), int(date.split('/')[2]), int(time.split(':')[0]), int(time.split(':')[1]), int(time.split(':')[2]))
			lat = float(row['Lat'])
			lng = float(row['Lng'])
			data.append(SPA_data(lat, lng, heading, d_time + timedelta(hours=-8)))
	return data

def main(args):
	fname = args.data_path
	if 30%int(args.sample_rate) != 0:
		raise AssertionError('(30/sample_rate) should be divisible!')
	interval = int(30/int(args.sample_rate))
	save_name = os.path.basename(fname).split('.')[0] + '_lightData.csv'
	print('save lighting data as:', save_name)
	SPA_data = readSPAData(fname)
	
	with open(save_name, 'w' , newline='') as f:
		fieldnames = ['---','isVisible','direction','RGB']
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(SPA_data) - 1):
			theta, phi = SPA_data[i].get_sunpos()
			theta_2, phi_2 = SPA_data[i+1].get_sunpos()
			for j in range(interval):
				direction = "(X=0.000000,Y="+ str("%.6f" % (phi * (1 - j/interval) + phi_2 * (j/interval))) + ",Z="+ str("%.6f" % (theta * (1 - j/interval) + theta_2 * (j/interval))) + ")"
				writer.writerow({'---' : i*interval + j,'isVisible': 'TRUE', 'direction': direction, 'RGB' : "(X=1.000000,Y=1.000000,Z=1.000000)" })
			if(i == len(SPA_data) - 2):
				theta, phi = SPA_data[i+1].get_sunpos()
				direction = "(X=0.000000,Y="+ str("%.6f" % phi) + ",Z="+ str("%.6f" % theta) + ")"
				writer.writerow({'---' : (i+1)*interval,'isVisible': 'TRUE', 'direction': direction, 'RGB' : "(X=1.000000,Y=1.000000,Z=1.000000)" })
        

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	# Path related arguments
	parser.add_argument('--data_path', required=True)
	parser.add_argument('--sample_rate', default=10, help='The sample rate of GPS data, default is 10 GPS per second')
	
	args = parser.parse_args()
	main(args)