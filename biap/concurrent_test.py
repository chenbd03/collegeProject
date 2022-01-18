# import os
# import tornado
# import tornado.ioloop
# import tornado.web
# import requests
# from concurrent.futures import ThreadPoolExecutor
# from tornado import gen
# import time
 
# class MainHandler(tornado.web.RequestHandler):
#     @tornado.web.asynchronous
#     def get(self):
#         self.write("Hello, world")
#         self.finish()

# class NoBlockingHnadler(tornado.web.RequestHandler):
#     @gen.coroutine
#     def get(self):
#         print('not blocking')
#         yield gen.sleep(10)
#         self.write('Blocking Request')


# class BlockingHnadler(tornado.web.RequestHandler):
#     def get(self):
#         print('blocking')
#         time.sleep(10)
#         self.write('Blocking Request')

# class WebServerApplication(object):
#     def __init__(self, port):
#         self.port = port
#         self.settings = {'debug': True}
 
#     def make_app(self):
#         """ 构建Handler
#         (): 一个括号内为一个Handler
#         """
 
#         return tornado.web.Application([
#             (r"/hello?", MainHandler),
#             (r"/noblock?", NoBlockingHnadler),
#             (r"/block?", BlockingHnadler),

#         ], ** self.settings)
 
#     def process(self):
#         """ 构建app, 监听post, 启动服务 """
 
#         app = self.make_app()
#         app.listen(self.port)
#         tornado.ioloop.IOLoop.current().start()
 
 
# if __name__ == "__main__":
#     # 定义服务端口
#     server_port = "10001"
#     server = WebServerApplication(server_port)
#     print('running')

#     server.process()

# import tornado.ioloop
# import tornado.web
# import tornado.httpserver
# import tornado.gen
# import time
# import datetime as dt


# class MainHandler(tornado.web.RequestHandler):
#     @tornado.gen.coroutine
#     def get(self):
#         yield tornado.gen.sleep(5)
#         self.write(str(dt.datetime.now()))

# def make_app():
#     return tornado.web.Application([
#         (r"/", MainHandler),
#     ], debug=True)

# if __name__ == "__main__":
#     app = make_app()
#     http_server = tornado.httpserver.HTTPServer(app)
#     http_server.listen(8000)
#     print('running')
#     tornado.ioloop.IOLoop.instance().start()
#     
#     
# import tornado.web
# import tornado.ioloop
# import tornado.httpserver   #tornado的HTTP服务器实现
# import time
# import datetime as dt 

# class IndexHandler(tornado.web.RequestHandler):
#     def get(self):      #处理get请求的，并不能处理post请求
#         time.sleep(5)
#         self.write(str(dt.datetime.now()))      #响应http的请求

# if __name__=="__main__":

#     app = tornado.web.Application([
#             (r"/" ,IndexHandler),
#         ])
#     httpServer = tornado.httpserver.HTTPServer(app)

#     #httpServer.listen(8000)  #多进程改动原来的这一句

#     httpServer.bind(8888)      #绑定在指定端口
#     httpServer.start(1)
#                                 #默认（0）开启一个进程，否则对面开启数值（大于零）进程
#                           #值为None，或者小于0，则开启对应硬件机器的cpu核心数个子进程
#                             #例如 四核八核，就四个进程或者八个进程
#     tornado.ioloop.IOLoop.current().start()


import os
import tornado
import tornado.ioloop
import tornado.web
import requests
from concurrent.futures import ThreadPoolExecutor
 
 
class Executor(ThreadPoolExecutor):
    """ 创建多线程的线程池，线程池的大小为10
    创建多线程时使用了单例模式，如果Executor的_instance实例已经被创建，
    则不再创建，单例模式的好处在此不做讲解
    """
    _instance = None
 
    def __new__(cls, *args, **kwargs):
        if not getattr(cls, '_instance', None):
            cls._instance = ThreadPoolExecutor(max_workers=10)
        return cls._instance
 
 
# 全部协程+异步线程池实现，yield在此的作用相当于回调函数
# 经过压力测试发现，此种方式的性能在并发量比较大的情况下，要远远优于纯协程实现方案
class Haha1Handler(tornado.web.RequestHandler):
    """ 获取域名所关联的IP信息 """
    # executor为RequestHandler中的一个属性，在使用run_on_executor时，必须要用，不然会报错
    # executor在此设计中为设计模式中的享元模式，所有的对象共享executor的值
    executor = Executor()
 
    @tornado.web.asynchronous  # 异步处理
    @tornado.gen.coroutine  # 使用协程调度
    def get(self):
        """ get 接口封装 """
 
        # 可以同时获取POST和GET请求参数
        value = self.get_argument("value", default=None)
 
        result = yield self._process(value)
        self.write(result)
 
    @tornado.concurrent.run_on_executor  # 增加并发量
    def _process(self, url):
        # 此处执行具体的任务
        try:
            resp = requests.get(url)
        except IOError as e:
            print(e)
            return 'failed'
 
        return 'success'
 
 
# 全部协程实现
class Haha2Handler(tornado.web.RequestHandler):
    """ 获取域名所关联的IP信息 """
 
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        """ get 接口封装 """
 
        # 可以同时获取POST和GET请求参数
        value = self.get_argument("value", default=None)
 
        result = yield tornado.gen.Task(self._process, value)
        self.write(result)
 
    @tornado.gen.coroutine  # 使用协程调度
    def _process(self, url):
        # 此处执行具体的任务
        try:
            resp = requests.get(url)
        except IOError as e:
            print(e)
            return 'failed'
 
        return 'success'
 
 
class WebServerApplication(object):
    def __init__(self, port):
        self.port = port
        self.settings = {'debug': False}
 
    def make_app(self):
        """ 构建Handler
        (): 一个括号内为一个Handler
        """
 
        return tornado.web.Application([
            (r"/gethaha1?", Haha1Handler),
            (r"/gethaha2?", Haha2Handler),
        ], ** self.settings)
 
    def process(self):
        """ 构建app, 监听post, 启动服务 """
 
        app = self.make_app()
        app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()
 
 
if __name__ == "__main__":
    # 定义服务端口
    server_port = "8888"
    server = WebServerApplication(server_port)
    server.process()




