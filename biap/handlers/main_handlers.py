#coding=utf-8

import tornado.web
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
import tornado.concurrent
import tornado.gen

from tornado.escape import json_encode,json_decode

import utils.common_util as common_util
import utils.result_view_util as result_util
# 导入人数检测模块
from libs.detect_c import detect
import time
import datetime as dt


class DetectFunction(object):
	def __init__(self):
		self.executor = ThreadPoolExecutor(10)   #不可以设为1或0

	@run_on_executor
	def face_count(self, img_data,img_name):
		img_arr = common_util.base64_2_array(img_data)
		detect_result = detect(images=img_arr,img_name=img_name)
		return detect_result


class FaceCountHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		# 设置请求头
		self.set_header('Content-Type', 'application/json')

	# @tornado.web.asynchronous
	@tornado.gen.coroutine
	def post(self):
		#print('报数了:', str(dt.datetime.now()))
		if not result_util.intercept_request_headers(self):
			return

		request_body = json_decode(self.request.body)
		if not result_util.is_valid_json(result_util.get_require_data(),request_body):
			result_util.write_error(self,'20004')
			return

		if common_util.valid_file_size(request_body['file']['data'],2):
			df = DetectFunction()
			detect_result= yield tornado.gen.maybe_future(df.face_count(request_body['file']['data'], request_body['file']['name']))
			# img_arr = common_util.base64_2_array(request_body['file']['data'])
			# detect_result = detect(images=img_arr)
			result_util.write_success(self,detect_result)
		else:
			result_util.write_error(self,'20006')

	def get(self):
		result_util.write_error(self, '20007')


class ErrorHandler(tornado.web.RequestHandler):
	def set_default_headers(self):
		# 设置请求头
		self.set_header('Content-Type', 'application/json')

	def get(self,_):
		# self.set_statues(404)
		result_util.write_error(self,'20009')


