import argparse
import sys
from sun_estimate.PSP import *
from misc.visual import writeVideo

def main(args):
	
	segmentation_module = buildModule(args)
	#frames = readVideo(1, fname = args.video_path)
	l = []
	temp_im = cv2.cvtColor(np.array(cv2.imread('0lqnDEkp1cW0gqCvs8mx-g.png')), cv2.COLOR_BGR2RGB)
	ims = setData(temp_im)
	l.append(Key_frames(ims))
	seg_list = inference(args, l, segmentation_module, seg_only=True)
	cv2.imwrite('seg.png', seg_list[0])
	#writeVideo('../out_seg.mp4',seg_list)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	# Path related arguments
	#parser.add_argument('--video_path', required=True)
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