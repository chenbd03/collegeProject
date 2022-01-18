#coding=utf-8

settings = dict(
	static_path='static',
	debug=True,
	# xsrf_cookies=True,
)

# settings = dict(
#         template_path='templates', #设置模板路径
#         static_path='static', #设置静态文件路径
#         debug=True, #调试模式
#         cookie_secret='aaaa', #cookie加密方式
#         login_url='/auth/user_login', #auth  指定默认的路径
#         xsrf_cookies=True, #防止跨域攻击
#         #ui_methods=admin_uimethods,
#         #pycket配置信息
#         pycket={
#             'engine': 'redis',
#             'storage': {
#                 'host': 'localhost',
#                 'port': 6379,
#                 'db_sessions': 5,
#                 'db_notifications': 11,
#                 'max_connections': 2 ** 31,
#             },
#             'cookies': {
#                 'expires_days':30, #设置过期时间
#                 #'max_age':5000,
#             }
#         }
# )