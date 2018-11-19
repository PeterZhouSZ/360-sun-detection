import cv2
import glob
import os
import csv
import json
import progressbar

file_list = []
#for filename in glob.glob('..\..\GSV_validate\images\*.png'):
for filename in glob.glob('D://360 google\Google_Street_View_Full_2/*png'):
    file_list.append(filename)


with open('D://360 google/GS_list_2.csv', 'w' , newline='') as f:
	fieldnames = ['filename','Y','M','lat','lng','yaw','pitch']
	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()
	for i in progressbar.progressbar(range(len(file_list))):
		f_name = os.path.splitext(os.path.basename(file_list[i]))[0] 
		if(cv2.imread(file_list[i]) is not None):
			img = cv2.resize(cv2.imread(file_list[i]), (1024, 512))
			#save_name = '..\..\GSV_validate\images_1024/' + f_name + '.png'
			save_name = 'D://360 google\Google_Street_View_1024_2/' + f_name + '.png'
			#json_data = json.loads(open('..\..\GSV_validate\data/'+ f_name + '.json',  encoding = 'utf8').read())
			if(os.path.isfile('D://360 google\GSV_data_2/'+ f_name + '.json')):
				cv2.imwrite(save_name, img)
				json_data = json.loads(open('D://360 google\GSV_data_2/'+ f_name + '.json',  encoding = 'utf8').read())
				writer.writerow({
					'filename' : save_name, 
					'Y': int(json_data['Data']['image_date'][:4]),
					'M': int(json_data['Data']['image_date'][5:7]),
					'lat': json_data['Location']['lat'],
					'lng': json_data['Location']['lng'],
					'yaw': json_data['Projection']['pano_yaw_deg'],
					'pitch': json_data['Projection']['tilt_pitch_deg']})
	