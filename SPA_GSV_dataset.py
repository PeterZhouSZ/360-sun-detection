import sys
import math
import csv
import cv2
import argparse
import numpy as np
import math
import os
import progressbar
from sun_estimate.light_detect import *
from sun_estimate.PSP import *
from datetime import datetime, timedelta
from misc.visual import drawImage, getImagePos
from misc.sunposition import sunpos, observed_sunpos

class GSV_data(object):
    def __init__(self, filename, y, m, lat, lng, yaw, pitch):
        self.f_name = filename
        self.year = y
        self.month = m
        self.lat = lat
        self.lng = lng
        self.yaw = yaw
        self.pitch = pitch

def readDataset(fname):
	data_list  = []
	with open(fname, newline='') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			panoid = os.path.splitext(os.path.basename(row['filename']))[0]
			data_list.append(GSV_data(row['filename'], row['Y'], row['M'], row['lat'], row['lng'], row['yaw'], row['pitch']))
	return data_list

def get_sunpos(time, lat, lng, yaw, pitch):
    #print(self.LAT, self.LNG)
    az,al = sunpos(time, lat, lng, 0)[:2]
    altitude = - (90 - al)
    azimuth = az - yaw - 90
    if(azimuth < -270):
        azimuth = 90 - (-270 - azimuth)
    return [azimuth, altitude]


def main(args):
	
	for cluster in range(1):
		data_list = readDataset(args.list_path)

		# img_list is prepared for segmentation module
		print('Loading images...')
		img_list = []
		est_list = []
		for i in progressbar.progressbar(range(len(data_list))):
			img = cv2.imread(data_list[i].f_name)
			img = np.array(img[:,:,:3], np.uint8)
			#cv2.imwrite('test'+str(i)+'.png', img)
			img_list.append(Key_frames(cv2.resize(img, (512, 256)))) # resolution should be 512x256
		
		segmentation_module = buildModule(args)
		seg_list = inference(args, img_list, segmentation_module, seg_only = True)
		
		#construct equir-array
		equir = np.zeros((256,512,3))
		for i in range(256):
			for j in range(512):
				t = Convert2Spherical(512, 256, j, i)
				t2 = Convert2Vector(t[0], t[1])
				equir[i,j,:] = t2

		result_list = []
		count = 0

		for i in progressbar.progressbar(range(len(data_list))):
			pano_id = os.path.splitext(os.path.basename(data_list[i].f_name))[0]
			# save segment result
			#cv2.imwrite('../../semantic/' + pano_id + '_seg.png', seg_list[i])
			# needed information in SPA		
			yaw = float(data_list[i].yaw)
			pitch = float(data_list[i].pitch)
			lat = float(data_list[i].lat)
			lng = float(data_list[i].lng)
			# Calculate all possible SP in one specific day
			SPA_list = []
			sunrise = datetime(int(data_list[i].year), int(data_list[i].month), 15, 4, 30, 0, 0) + timedelta(hours=-8) 
			for h in range(15):
				tmp_t = sunrise + timedelta(hours=h)
				for m in range(6):
					SP = get_sunpos( tmp_t + timedelta(minutes=10*m), lat, lng, yaw, pitch)
					SPA_list.append(SP)
			# Find the most reliable SP, remember to change BGR format to RGB
			final_sp, confidence, index = SunDetection_SPA(img_list[i].im, seg_list[i], SPA_list, i, equir)
			#final_sp, confidence, index = SunDetection_SPA(img_list[i].im, cv2.imread('../../semantic/' + str(i) + '_seg.png'), SPA_list, i, equir)
			is_label = False
			if(confidence < -1000.0):
				#cv2.imwrite( '../../results/discard/small_sky/' + str(i) + '.png', img_list[i].im)
			else:
				if(confidence >= 0.7):
					is_label = True
					result = drawImage(img_list[i].im, final_sp[0], final_sp[1], (0, 0, 255))
					#cv2.imwrite('../../results/label/' + str(i) + '.png', result)
					result_list.append([final_sp[0], final_sp[1], i, confidence, index])
			'''			
			if(is_label):
				frame = cv2.resize(cv2.imread('D://360 google\Google_Street_View_Full/' + pano_id + '.png'), (1664, 832))
				cv2.imwrite('../../label_data/image_sequence/pano_' + str("%04d" % count) + '.png', frame)
				count += 1
				with open('../../label_data/data/' + pano_id + '.csv', 'w', newline = '') as f:
					fieldnames = ['temporal', 'y', 'z']
					writer = csv.DictWriter(f, fieldnames=fieldnames)
					writer.writeheader()
					for i in range(len(SPA_list)):
						writer.writerow({'temporal' : i, 'y' : SPA_list[i][1], 'z' : SPA_list[i][0]})
			'''
		
		# write data to CSV
		with open('GS_sunpos.csv', 'w' , newline='') as f:
			fieldnames = ['count','panoID', 'theta', 'phi', 'confidence', 'index']
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for i in range(len(result_list)):
				pano_id = os.path.splitext(os.path.basename(data_list[result_list[i][2]].f_name))[0]
				sunpos = Convert2Spherical(512, 256, result_list[i][0], result_list[i][1])
				writer.writerow({'count' : i, 'panoID' : pano_id, 'theta' : sunpos[0], 'phi' : sunpos[1], 'confidence' : result_list[i][3], 'index' : result_list[i][4]})

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--list_path', required=True)
	parser.add_argument('--model_path', default='sun_estimate/pre-trained',
						help='folder to model path')
	parser.add_argument('--suffix', default='_epoch_20.pth',
						help="which snapshot to load")

	# Model related arguments
	parser.add_argument('--arch_encoder', default='resnet50_dilated8',
						help="architecture of net_encoder")
	parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
						help="architecture of net_decoder")
	parser.add_argument('--fc_dim', default=2048, type=int,
						help='number of features between encoder and decoder')

	# Data related arguments
	parser.add_argument('--num_val', default=-1, type=int,
						help='number of images to evalutate')
	parser.add_argument('--num_class', default=150, type=int,
						help='number of classes')
	parser.add_argument('--batch_size', default=1, type=int,
						help='batchsize. current only supports 1')
	parser.add_argument('--imgSize', default=[300, 400, 500, 600],
						nargs='+', type=int,
						help='list of input image sizes.'
							 'for multiscale testing, e.g. 300 400 500')
	parser.add_argument('--imgMaxSize', default=1000, type=int,
						help='maximum input image size of long edge')
	parser.add_argument('--padding_constant', default=8, type=int,
						help='maxmimum downsampling rate of the network')
	parser.add_argument('--segm_downsampling_rate', default=8, type=int,
						help='downsampling rate of the segmentation label')

	# Misc arguments
	parser.add_argument('--result', default='.',
						help='folder to output visualization results')
	parser.add_argument('--gpu_id', default=0, type=int,
						help='gpu_id for evaluation')

	args = parser.parse_args()
	#print(args)

	# torch.cuda.set_device(args.gpu_id)

	# absolute paths of model weights
	args.weights_encoder = os.path.join(args.model_path,
										'encoder' + args.suffix)
	args.weights_decoder = os.path.join(args.model_path,
										'decoder' + args.suffix)

	assert os.path.exists(args.weights_encoder) and \
		os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

	if not os.path.isdir(args.result):
		os.makedirs(args.result)

	main(args)