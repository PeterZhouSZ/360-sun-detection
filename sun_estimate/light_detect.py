import sys
import cv2
from skimage import measure
from skimage import filters
import math
import numpy as np
from matplotlib import pyplot as plt
from misc.visual import getImagePos, drawImage

def Convert2Vector(theta, phi):
	return np.array([math.cos(math.radians(phi)) * math.sin(math.radians(theta)), math.sin(math.radians(phi)), math.cos(math.radians(phi)) * math.cos(math.radians(theta))])
def world2sphere(x, y, z):
    u = (1 / np.pi) * np.arctan2(x, z)
    v = (1 / np.pi) * np.arcsin(y)
    u = u / 2
    return np.array([u * 360.0, v * 180.0])
def Convert2UV(width, height, theta, phi):
	x = theta/180
	y = phi/90
	i = (x + 1.0) * width/2
	j = (1 - (y + 1.0)/2)*height
	return np.array([i,j])
def Convert2Spherical(width, height, i, j):
	x = (i / (width/2)) - 1.0 
	y = -(j - (height/2))/(height/2)
	theta = x * 180
	phi =  y * 90
	return np.array([theta, phi])
def Theta2XY(theta):
    x = math.sin(math.radians(theta))
    y = math.cos(math.radians(theta))
    return np.array([x,y])
def XY2Theta(XY):
    if(XY[1] > 0):
        theta = math.degrees(math.asin(XY[0]))
    elif (XY[1] < 0 and XY[0] < 0):
        theta = -math.degrees(math.acos(XY[1]))
    else: 
        theta = math.degrees(math.acos(XY[1]))
    return theta
def getAngle(width, height, s, t):
	x = Convert2Spherical(width, height, s[0], s[1])
	y = Convert2Spherical(width, height, t[0], t[1])
	X = Convert2Vector(x[0], x[1])
	Y = Convert2Vector(y[0], y[1])
	return np.dot(X,Y)

def visualAll(img, spa_list, final_sp, mask):
	for i in range(len(spa_list)):
		coord = getImagePos(spa_list[i][1], spa_list[i][0], width=512, height=256)
		img = drawImage(img, coord[0], coord[1], (0,255,0))
	img = drawImage(img, final_sp[0], final_sp[1], (0, 0, 255))
	m = np.full((mask.shape[0], mask.shape[1], 3), (0, 0, 255))
	final = m * 0.3 + img * 0.7
	i, j = np.where(mask == False)
	final[i,j,:] = img[i,j,:]
	return final

def SunDetection_SPA(img, seg, SPA_list, id, EQ):
	'''
	Input resolution should be 512x256 
	This detection method only be used in the image with GPS and time (Y/M) information
	Warning! It returns the image position in x/y axis
	'''
	ori_img = img.copy() # opencv BGR format
	img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	img = img[:, :, ::-1]
	# Get the luminance
	luminance = np.dot(img[:,:,0:3],[0.3,0.59,0.11])/255.0
	luminance = img_lab[:,:,0]/255.0
	#print( ' id : ' + str(id))
	sky_color = np.array([230, 230, 6]) #seg is BGR , sky = (230, 230, 6)
	sky_index = np.where(np.all(seg == sky_color, axis = -1))
	sky_seg = np.zeros(luminance.shape)
	sky_seg[sky_index] = 1.0
	sky_seg[128:256,:] = 0.0
	# filter out the SPA SP which is not locate in sky region

	total_pixels = sky_seg.sum()
	if((total_pixels/(256*256)) < 0.4):
		return None, -100000, None
	luminance = cv2.blur(luminance, (5,5))
	sort_sky = np.sort(luminance[sky_index], axis = None)
	threshold = sort_sky[int(total_pixels * 0.80)] #threshold
	light_patches = (luminance >= threshold) * sky_seg
	#plt.imsave(str(id) + '.png', light_patches)
	#light_pixels = light_patches.sum()
	
	'''
	# rank the connected component and get top n light patches
	connected_component = measure.label(light_patches, connectivity=2, background = 0)
	# sort all light patches
	patch_list = []
	num = connected_component.max()
	for i in range(1, num + 1):
		size = len(np.where(connected_component == i)[0])
		patch  = (i, size)
		patch_list.append(patch)
	dtype = [('label', int), ('size', int)]
	patches = np.array(patch_list,dtype = dtype)
	sorted_patches = np.sort(patches, order = 'size')
	'''
	max_confidence = -1000.0
	sun_position = None
	sun_temporal = None
	for i in range(len(SPA_list)):
		coord = getImagePos(SPA_list[i][1], SPA_list[i][0], width=512, height=256)
		if(sky_seg[int(coord[1]),int(coord[0])] == True):
			target = Convert2Spherical(512, 256, coord[0], coord[1])
			tmp = np.dot(EQ, Convert2Vector(target[0], target[1]))
			mask = (tmp > 0.82) * light_patches
			confidence = (tmp * luminance * mask).sum()
			if(confidence > max_confidence):
				max_confidence = confidence
				sun_position = coord
				sun_temporal = i

	score = max_confidence/(luminance * light_patches).sum()
	#print( '   ' + str(id) + '  :  ' + str(score) + '....')
	if(max_confidence > -1000.0):
		visual = visualAll(ori_img, SPA_list, sun_position, light_patches)
		#cv2.imwrite('../../visualize/' + str(id) + '.png', visual)
	return sun_position, score, sun_temporal


def SunDetection_seg(img, seg, id):
	'''
	Baseline sun position detection:
	Input size: 512x256
	1. Fetch sky region by segmentation result (assume that sky only be in top half region)
	2. Use 98th highest luminance value for thresholding (refer to DOIE)
	3. =
	'''
	ori_img = img
	#print(img.shape, seg.shape)
	# Get the luminance and HDR radiance
	luminance = np.dot(img[:,:,0:3],[0.3,0.59,0.11])

	sky_color = np.array([230, 230, 6]) #seg is BGR , sky = (230, 230, 6)
	sky_index = np.where(np.all(seg == sky_color, axis = -1))
	sky_seg = np.zeros(luminance.shape)
	sky_seg[sky_index] = 1.0
	total_pixels = sky_seg[0:int(img.shape[0]/2),:].sum()
	sky =  sky_seg * luminance
	#sky *= 255.0
	total_light = sky.sum()
	sort_sky = np.sort(sky, axis = None)
	threshold = sort_sky[int(sky.size * 0.98)]
	threshold_img = sky >= threshold
	light_patch = np.ones(luminance.shape) * threshold_img
	#plt.imsave(str(id)+'.png', sky_seg*50.0 + light_patch * 50.0)
	light_pixels = threshold_img[0:int(img.shape[0]/2),:].sum()
	#print(light_pixels)
	
	if(light_pixels > (total_pixels * 0.8) or light_pixels == 0):
		#print("un-visible!")
		return False, 0.0, -90.0, 0.0, 1.0, 1.0, 1.0
	else:	 
		# rank the connected component and get top n light patches
		connected_component = measure.label(threshold_img, connectivity=2, background = 0)
		# sort all light patches
		patch_list = []
		num = connected_component.max()
		for i in range(1, num + 1):
			size = len(np.where(connected_component == i)[0])
			patch  = (i, size)
			patch_list.append(patch)
		dtype = [('label', int), ('size', int)]
		patches = np.array(patch_list,dtype = dtype)
		sorted_patches = np.sort(patches, order = 'size')
		
		label = connected_component == sorted_patches[num - 1]['label']
		threshold_img = label != 0

		irr_sum = 0.0
		irr_XY_sum = np.array([0.0,0.0])
		irr_phi_sum = 0.0
		for y in range(int(threshold_img.shape[0]/2)): 
			for x in range(threshold_img.shape[1]): 
				if(threshold_img[y,x] == True):
					angle = Convert2Spherical(ori_img.shape[1],ori_img.shape[0],x,y)
					v = Theta2XY(angle[0])
					irr_sum += math.cos(math.radians(angle[1]))
					irr_XY_sum += v * math.cos(math.radians(angle[1]))
					irr_phi_sum += angle[1] * math.cos(math.radians(angle[1]))

		final_XY = irr_XY_sum/irr_sum
		norm = np.linalg.norm(final_XY)
		final_XY = final_XY/norm
		final_angle = np.array([XY2Theta(final_XY), irr_phi_sum/irr_sum])
		light_y = final_angle[1]
		light_z = final_angle[0]
		if(light_z < -180.0):
			light_z += 360

		sun_position = Convert2UV(ori_img.shape[1],ori_img.shape[0], final_angle[0], final_angle[1])
		
		y = int(round(sun_position[1],0))
		x = int(round(sun_position[0],0))
		if(x >= ori_img.shape[1]):
			x = 0
		R = ori_img[y, x, 0] 
		G = ori_img[y, x, 1] 
		B = ori_img[y, x, 2] 
		return True, 0.000000, "%.6f" % light_y, "%.6f" % light_z, "%.6f" % R, "%.6f" % G, "%.6f" % B