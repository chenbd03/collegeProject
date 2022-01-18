#coding=utf-8

import tornado.web
from tornado.options import define, options
from config import settings
from handlers.main_urls import handlers 

#定义一个默认的接口
define("port", default=88, help="run port",type=int)


def run():
	#创建应用实例
	options.parse_command_line()
	#通过应用实例创建服务器实例
	app = tornado.web.Application(
		handlers,
		**settings
	)
	http_server = tornado.httpserver.HTTPServer(app)
	http_server.listen(options.port)
	print('application run in port {}'.format(options.port))
	tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
	run()
