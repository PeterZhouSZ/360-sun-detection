import argparse
import numpy as np
import cv2
import math
import csv
from misc.visual import drawImage, writeVideo, getImagePos
import progressbar
	
def readCSV(fname):
    light_data  = []
    with open(fname, newline='') as csvfile:

        rows = csv.DictReader(csvfile)

        for row in rows:
            light_data.append([float(row['direction'][14:22]), float(row['direction'][27:35])])
    return light_data

def readGT(fname):
	GT = []	
	with open(fname, newline='') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			GT.append([float(row['phi']), float(row['theta'])])
	return GT

def main(args):
	fname = args.video_path

	data_len = 0
	if(args.data_path != '0'):
		pos_list = readCSV(args.data_path)
		data_len = len(pos_list)

	frames = []
	cap = cv2.VideoCapture(fname)

	print('Start processing visualize data')
	for i in progressbar.progressbar(range(data_len)):
		success, frame = cap.read()
		if(success):
			frame = cv2.resize(frame, (1024, 512))
			img = frame[:,:,:3]
			#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = np.array(img, np.uint8)

			pos = pos_list[i]
			p = getImagePos(1024, 512, pos[0], pos[1])
			img = drawImage(img, p[0], p[1], (0, 0, 255))		
			frames.append(img)

	print('Writing video...')	
	writeVideo('out.mp4',frames)
	print('Done!')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	# Path related arguments
	parser.add_argument('--video_path', required=True)
	parser.add_argument('--data_path', required=True, help = 'Estimated sun position data')
	
	args = parser.parse_args()
	print(args)

	main(args)
