
from models.result_model import code_msg
from tornado.escape import json_encode
from utils.common_util import *


def get_require_data():
	request_type = {
		'device': {
			'model': False,
			'plat': False,
			'version': False,
		},
		'file': {
			'type': False,
			'data': False,
		},
		'person_id': False,
		'request_id': False,
	}
	return request_type

def get_result_model():

	return 	{
				'code': '',
				'msg': '',
			    'request_id': '0',
			    'data': {},
			    'response_id': '0',
			    'cost_time': 0
			}


def is_valid_json(require_data,data):
	tag = True
	for key in require_data:
		if key in data.keys():
			if require_data[key] != False:
				tag = is_valid_json(require_data[key], data[key])
		else:
			return False
		if not tag:
			return False
	return True

def intercept_request_headers(self):
	headers = self.request.headers
	if not valid_require_headers(headers):
		write_error(self, '20003')
		return False
	if not valid_appid(headers.get('X-Appid')):
		write_error(self, '20010')
		return False
	if not valid_timestamp(headers.get('X-Timestamp')):
		write_error(self, '20008')
		return False
	if not valid_authorization(headers.get('X-Authorization')):
		write_error(self, '20002')
		return False
	return True

	
def write_success(self, data):
	result = get_result_model()
	result['code'] = '0'
	result['msg'] = code_msg['0']
	result['data'] = data
	if 'cost_time' in data.keys():
		result['cost_time'] = data.pop('cost_time')
	self.write(json_encode(result))


def write_error(self, status_code):
	result = get_result_model()
	result['code'] = status_code
	result['msg'] = code_msg[status_code]
	self.write(json_encode(result))




