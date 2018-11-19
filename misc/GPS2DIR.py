import math
import csv
import argparse

def readGPS(fname):
    GPS_data  = []
    with open(fname, newline='') as csvfile:

        rows = csv.DictReader(csvfile)

        for row in rows:
            GPS_data.append([float(row['lat']), float(row['lng'])])
    return GPS_data

def GPS2Dir(gps):
    dir_data = []   
    for i in range(len(gps) - 1):
        y = math.sin(gps[i+1][1] - gps[i][1]) * math.cos(gps[i+1][0])
        x = math.cos(gps[i][0]) * math.sin(gps[i+1][0]) - math.sin(gps[i][0]) * math.cos(gps[i+1][0]) * math.cos(gps[i+1][1] - gps[i][1])
        brng = round(math.degrees(math.atan2(y, x)))
        dir_data.append(brng)
        if (i ==  (len(gps) - 2)):
            dir_data.append(brng)
    return dir_data

def writeDir(g, d):
    with open('data.csv', 'w' , newline='') as f:
        fieldnames = ['---','lat','lng','dir']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(g)):    
            writer.writerow({'---' : i, 
                             'lat': g[i][0], 
                             'lng': g[i][1], 
                             'dir': d[i]})

def main(args):
    fname = args.data_path
    GPS = readGPS(fname)
    DIR = GPS2Dir(GPS)
    writeDir(GPS, DIR)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	# Path related arguments
	parser.add_argument('--data_path', required=True)
	
	args = parser.parse_args()

	main(args)