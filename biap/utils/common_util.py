#coding=utf-8
import cv2
import numpy as np
import base64
import time

def get_require_headers():
	return {'X-Appid', 'X-Deviceid', 'X-Timestamp',  'X-Authorization'}

def valid_require_headers(headers):
	require_headers = get_require_headers()
	for key in require_headers:
		if not key in headers:
			return False
	return True

def valid_appid(appid):
	return True

def valid_timestamp(timestamp, limited_time=10):
	timestamp = int(timestamp)
	current = int(time.time()*1000)
	#print(current-timestamp)
	#print(limited_time*60000)
	#print(current-timestamp >= limited_time*60000)
	if current-timestamp >= limited_time*60000:
		return False
	return True

def valid_authorization(authorization):
	return True

def valid_file_size(base64_data, limited_size):
	data = base64_data[:base64_data.find('=')]
	data_len = len(data)
	file_size = data_len-((data_len/8) * 2)

	return (file_size/1024/1024) <= limited_size


def base64_2_array(base64_data):
    im_data = base64.b64decode(base64_data)
    im_array = np.frombuffer(im_data, np.uint8)
    im_array = cv2.imdecode(im_array, cv2.COLOR_RGB2BGR)
    return im_array

