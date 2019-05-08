'''
    Some function for visualization
'''
import numpy as np
import cv2

def getImagePos(i, j, width=1024, height=512): 
	'''
	UE4 vector (y and z) to image position
	i : y
	j : z
	'''
	theta = j + 90.0
	if(theta > 180.0):
		theta = theta - 360.0
	phi = -i
	x = theta/180
	y = phi/90
	return np.array([(x + 1.0) * width/2, (1 - (y + 1.0)/2)*height])

def drawImage(img, x, y, color):
    r = 10
    pts = np.array([[x,y - r], [x - r, y + 1/2 * r] , [x + r, y + 1/2 * r]],np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img, [pts], color)
    pts = np.array([[x,y + r], [x - r, y - 1/2 * r] , [x + r, y - 1/2 * r]],np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img, [pts], color)
    return img

def writeVideo(name, frames):
	frame_width = frames[0].shape[1]
	frame_height = frames[0].shape[0]
	out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width,frame_height))
	for i in range(len(frames)):
		#cv2.imwrite('out' + str(i) + '.png', frames[i])
		out.write(frames[i])
	out.release() 