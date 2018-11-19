# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
import numpy as np
import math
import csv
import progressbar
from sun_estimate.PSP import *
from sun_estimate.light_detect import Convert2Vector, world2sphere

def main(args):
	
	segmentation_module = buildModule(args)

	save_name = os.path.basename(args.video_path).split('.')[0] + '_lightData.csv'
	# Load whole video into memory
	frames = readVideo(args.interval, fname = args.video_path)

	IBS_data = inference(args, frames, segmentation_module)

	'''
		Write csv lighting data
	'''
	print("Write IBS Data to CSV file")
	with open(save_name, 'w' , newline='') as f:
		fieldnames = ['---','isVisible','direction','RGB']
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		frame_num = len(IBS_data) - 1 #total predictions - 1

		if(args.interval == 1):
			for i in range(frame_num):    
				if(IBS_data[i].isVisible == False):
					writer.writerow({'---' : str(i), 
									'isVisible': 'FALSE' , 
									'direction' : "(X=0.000000,Y=-90.000000,Z=0.000000)" , 
									'RGB' : "(X=1.000000,Y=1.000000,Z=1.000000)" })
				else:
					light_color = IBS_data[i].RGB
					position = IBS_data[i].position
					writer.writerow({'---' : str(i), 
									'isVisible': 'TRUE' , 
									'direction' : "(X=" + str("%.6f" % position[0]) + ",Y="+ str("%.6f" % (-position[1])) + ",Z="+ str("%.6f" % (position[2] - 90.0)) + ")" , 
									'RGB' : "(X=1.000000,Y=1.000000,Z=1.000000)" })
		else:
			for i in range(frame_num):    
				if(IBS_data[i].isVisible == False or IBS_data[i+1].isVisible == False):
					writer.writerow({'---' : str(i * args.interval), 
									'isVisible': 'FALSE' , 
									'direction' : "(X=0.000000,Y=-90.000000,Z=0.000000)" , 
									'RGB' : "(X=1.000000,Y=1.000000,Z=1.000000)" })
				else:
					for j in range(args.interval):
						light_color = IBS_data[i].RGB + (IBS_data[i+1].RGB - IBS_data[i].RGB) * (j/args.interval)
						v1 = Convert2Vector(IBS_data[i].position[2], IBS_data[i].position[1])
						v2 = Convert2Vector(IBS_data[i+1].position[2], IBS_data[i+1].position[1])
						v = v1 + (v2 - v1) * (j/args.interval)
						v_final = v/np.linalg.norm(v)
						final_pos = world2sphere(v_final[0], v_final[1], v_final[2])
						writer.writerow({'---' : str(i * args.interval + j), 
										'isVisible': 'TRUE' , 
										'direction' : "(X=0.000000,Y=" + str("%.6f" % (-final_pos[1])) + ",Z="+ str("%.6f" % (final_pos[0] - 90.0)) + ")" , 
										'RGB' : "(X=1.000000,Y=1.000000,Z=1.000000)" })
	print('Done!')	

if __name__ == '__main__':
	assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
		'PyTorch>=0.4.0 is required'

	parser = argparse.ArgumentParser()
	# Path related arguments
	parser.add_argument('--video_path', required=True)
	parser.add_argument('--interval', default=30, type=int, help='interval between two key frames')
	parser.add_argument('--model_path', default='pre-trained',
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
