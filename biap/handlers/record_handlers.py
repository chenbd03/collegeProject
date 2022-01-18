
from tornado.options import define, options
import tornado.web
import utils.result_view_util as result_util 

define("visits", default=83)

class VisitRecordHandler(tornado.web.RequestHandler):
	
	def get(self):
		options.visits = options.visits + 1
		data = {
			'visits' : options.visits
		}
		result_util.write_success(self, data)
