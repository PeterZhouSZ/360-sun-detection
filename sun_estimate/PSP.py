# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
import cv2
from scipy.misc import imread, imresize
import math
import csv
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat
# Our libs
from sun_estimate.dataset import TestDataset
from sun_estimate.models import ModelBuilder, SegmentationModule
from sun_estimate.utils import colorEncode
from sun_estimate.lib.nn import user_scattered_collate, async_copy_to
from sun_estimate.lib.utils import as_numpy, mark_volatile
import sun_estimate.lib.utils.data as torchdata
from sun_estimate.light_detect import SunDetection_seg
import progressbar


class Key_frames(object):
	RGB = np.array([0.0,0.0,0.0])
	position = np.array([0.0,0.0,0.0])
	isVisible = True
	
	def __init__(self, ims):
		self.im = ims

def setData(input_img):
	im = cv2.resize(input_img, (512, 256))
	img = im[:,:,:3]
	return img

def readVideo(interval, fname = ''):
	print('Loading and processing all frames...')
	frames = []
	
	cap = cv2.VideoCapture(fname)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for i in progressbar.progressbar(range(length)):
		success, frame = cap.read()
		if(success):
			if(i % interval == 0):
				temp_im = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
				ims = setData(temp_im)
				frames.append(Key_frames(ims))
	print( 'Video preparation done!:' )
	print( 'Total:' + str(len(frames)) )

	return frames

def visualize_result(data, preds, args):
	colors = loadmat('sun_estimate/data/color150.mat')['colors']
	# prediction
	pred_color = colorEncode(preds, colors)
	return pred_color.astype(np.uint8)


def test(segmentation_module, loader, args, v_data, seg_only):
	segmentation_module.eval()
	seg = []
	with progressbar.ProgressBar(max_value=len(v_data)) as bar:
		for i, batch_data in enumerate(loader):
			# process data
			batch_data = batch_data[0]
			segSize = (batch_data['img_ori'].shape[0],
					batch_data['img_ori'].shape[1])

			img_resized_list = batch_data['img_data']

			with torch.no_grad():
				pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
				pred = Variable(pred).cuda()

				for img in img_resized_list:
					feed_dict = batch_data.copy()
					feed_dict['img_data'] = img
					del feed_dict['img_ori']
					del feed_dict['info']
					feed_dict = async_copy_to(feed_dict, args.gpu_id)

					# forward pass
					pred_tmp = segmentation_module(feed_dict, segSize=segSize)
					pred = pred + pred_tmp / len(args.imgSize)

				_, preds = torch.max(pred.data.cpu(), dim=1)
				preds = as_numpy(preds.squeeze(0))

			# visualization
			final_result = visualize_result(
				(batch_data['img_ori'], batch_data['info']),
				preds, args)
			
			if(seg_only):
				seg.append(final_result)
			else:
				#cv2.imwrite('seg' + str(i) + '.png' ,np.array(cv2.resize(final_result, (512, 256))))
				visibility, X, Y, Z, R, G, B = SunDetection_seg(v_data[i].im, np.array(cv2.resize(final_result, (512, 256))), str(i))
				v_data[i].RGB = np.array([R,G,B], dtype = 'float32')
				v_data[i].position = np.array([X,Y,Z], dtype = 'float32')
				v_data[i].isVisible = visibility
			bar.update(i)
			#print('[{}] iter {}'
			#	  .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i))
	if(seg_only):
		return seg
	return v_data

def buildModule(args):
	torch.cuda.set_device(args.gpu_id)

	# Network Builders
	builder = ModelBuilder()
	net_encoder = builder.build_encoder(arch=args.arch_encoder,
										fc_dim=args.fc_dim,
										weights=args.weights_encoder)
	net_decoder = builder.build_decoder(arch=args.arch_decoder,
										fc_dim=args.fc_dim,
										weights=args.weights_decoder,
										use_softmax=True)

	crit = nn.NLLLoss(ignore_index=-1)

	return SegmentationModule(net_encoder, net_decoder, crit)

def inference(args, frames, module, seg_only = False):

	list_test = []
	# pass all frames into test module
	for i in range(len(frames)):
		list_test.append({'fpath_img': frames[i].im})

	dataset_val = TestDataset(
		list_test, args, max_sample=len(frames))
	loader_val = torchdata.DataLoader(
		dataset_val,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=user_scattered_collate,
		num_workers=5,
		drop_last=True)

	module.cuda()		
	result = test(module, loader_val, args, frames, seg_only)
	return result

