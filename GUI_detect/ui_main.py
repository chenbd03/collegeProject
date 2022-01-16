from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from util import *

from yolov5.utils.datasets import LoadImages, LoadStreams, ReadOneImage
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
import os
import time
import datetime
import cv2
import random
import torch
import torch.backends.cudnn as cudnn

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from PyQt5 import QtChart
# import pyqtgraph as pg
from PyQt5 import QtWidgets
from GUI import Ui_MainWindow
import threading


import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# import sip


class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)


class Ui_main(QMainWindow, Ui_MainWindow, QWidget):
    def __init__(self):
        super(Ui_main, self).__init__()
        self.setupUi(self)
        self.classes = []   # 检测出来目标
        self.ImageIs = False  # 检测类型是否是静态图片
        self.WebcamIs = False # 是否是RTSP
        self.source = None
        
        self.img_size = 640
        self.dataset = None
        self.weights = "./yolov5/weights/yolov5s.pt"
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        
        self.label_name = ["行人", "汽车", "公交车", "自行车", "摩托车"]  # 检测目标总label
        self.label_risk = ["all_people", "l_risk", "h_risk"]
        self.detect_result_count = [0,0,0]  # 检测出来label_name中对应的数量

        self.c = 1
        self.n = 0
        
        # pg.setConfigOption('background', 'k')
        # pg.setConfigOption('foreground', 'w')

        self.pushButton_3.clicked.connect(self.get_detect_object_state)

        self.pushButton_4.clicked.connect(self.pictureStart)

        self.pushButton_6.clicked.connect(self.videoStart)

        self.pushButton.clicked.connect(self.get_rtspStream_str)

        self.pushButton_2.clicked.connect(self.clear_rtspStream_str)



        self.pushButton_5.clicked.connect(self.load_init_model)

        self.pushButton_7.clicked.connect(self.start_detect)

        self.timer_camera = QTimer()
        self.cap = cv2.VideoCapture()
        self.timer_camera.timeout.connect(self.show_camera)

        self.pushButton_8.clicked.connect(self.clear_Qlabel_monitor)

        # self.plot_table()

    def init_table_time(self):
        self.timer_table = QTimer()
        self.timer_table.timeout.connect(self.risk_people)
        self.timer_table.start(1000)
        

    def clear_Qlabel_monitor(self):
        self.timer_camera.stop()

        self.label_3.setPixmap(QPixmap(""))
        self.source = None
        if self.cap.isOpened() and not self.ImageIs:
            self.cap.release()  # 释放视频流
            self.printf("已释放视频流...")
        else:
            self.printf("已释放图片流...")

    def get_detect_object_state(self):
        self.classes = []
        if self.checkBox.isChecked():
            self.classes.append(0)
        if self.checkBox_2.isChecked():
            self.classes.append(2)
        if self.checkBox_3.isChecked():
            self.classes.append(5)
        if self.checkBox_4.isChecked():
            self.classes.append(1)
        if self.checkBox_5.isChecked():
            self.classes.append(3)
        
        self.printf("已确定检测目标...")
    # get_detect_object_state

    def pictureStart(self):
        self.WebcamIs = False
        pictureName, _ = QFileDialog.getOpenFileName(self.groupBox_3, "Open", "", "*.jpg;;*.png;;All Files(*)")
        if pictureName != "":
            self.source = pictureName
            self.ImageIs = True
            self.printf(self.source)
            self.printf("上传图片路径成功...")
            self.dataset = ReadOneImage(self.source)
            # self.dataset = LoadImages(self.source, img_size=self.img_size)
            self.printf("已读取成功...")
        else:
            self.source = None
            self.ImageIs = False

            self.printf("上传图片路径失败, 或被取消...")



    def videoStart(self):
        self.ImageIs = False
        self.WebcamIs = False
        pictureName, _ = QFileDialog.getOpenFileName(self.groupBox_4, "Open", "", "*.mp4;;*.avi;;All Files(*)")
        if pictureName != "":
            self.source = pictureName
            self.printf(self.source)
            self.printf("上传视频路径成功...")
            if not self.ImageIs:
                cfg = get_config()
                cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
                self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                    use_cuda=True)
            self.printf("已加载deepsort...")
        else:
            self.source = None
            self.printf("上传视频路径失败, 或被取消...")


    
    def get_rtspStream_str(self):
        """
        获取rtsp数据流的地址
        """
        self.ImageIs = False  # 将检测单张图片设置为False
        self.WebcamIs = True
        self.source = self.textEdit.text()

        self.printf("上传rtsp流成功...")
        if not self.ImageIs:   # 如果不是图片,则启动deepsort
            cfg = get_config()
            cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
            self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                    use_cuda=True)
        self.printf("已加载deepsort...")

    def clear_rtspStream_str(self):
        self.source = None

        self.textEdit.clear()
        self.dataset = None

        self.printf("已清除rtsp流地址...")

    def load_init_model(self):
        self.printf("请稍候, 正初始化程序...")
        self.half = True
        # Load model
        self.model,self.device = init_model(self.weights,self.half, self.img_size)
        self.half = self.device.type != 'cpu'

        self.printf("初始化成功...")

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        self.dataset = ReadOneImage(self.image)
        self.detect_one_image(self.dataset)
        self.c = self.c+1
        if self.c% int(self.frameRate/3) == 0:
            self.c = 1
            current_time = time.ctime()
            with open("inference/output/result.txt","a") as ff:
                ff.write(current_time+" current_all_people:"+str(self.n)+"\n")
        QtWidgets.QApplication.processEvents()

    def start_detect(self):
        """启动检测触发事件
        """
        if self.source is not None:
            if self.ImageIs:  
                self.detect_one_image(self.dataset)
            else:
                if self.source == "0":
                    self.source = int(self.source)
                
                if self.timer_camera.isActive() == False:  # 若定时器未启动
                    
                    flag = self.cap.open(self.source)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
                    if flag == False:  # flag表示open()成不成功/
                        pass
                    else:
                        self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
                        self.timer_camera.start(1.0 / self.frameRate)  # 定时器开始计时1ms，结果是每过1ms从摄像头中取一帧显示

                else:
                    self.timer_camera.stop()  # 关闭定时器
                    self.cap.release()  # 释放视频流
                    self.label_3.clear()  # 清空视频显示区域
        else:
            self.printf("未加载数据...")
            self.messageDialog("未找到图片或视频,请重新上传")


    def detect_one_image(self, img_data):
        """检测单张图片
        """
        self.detect_result_count.clear()
        img = torch.from_numpy(img_data.img).unsqueeze(0).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        im0 = img_data.img0
        pred = self.model(img, augment=False)[0]
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)[0]
        people_coords = []
        bbox_xywh = []
        confs = []
        low_risk_count = 0
        high_risk_count = 0
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
            self.n = n
            if self.ImageIs:
                for *xyxy, conf, cls in det:
                    people_coords.append(xyxy)
                draw_boxes(im0, people_coords, n)

            else:
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, im0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy,n,identities=identities)


            # draw_description(im0, n, low_risk_count, high_risk_count)
            image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            h,w, _ = image.shape
            qt_img = QImage(image.data,w,h,QImage.Format_RGB888)
            self.label_3.setScaledContents(True)
            self.label_3.setPixmap(QPixmap.fromImage(qt_img))

    
    def printf(self,mypstr):
        self.textBrowser.append(mypstr)   #在指定的区域显示提示信息
        self.cursor=self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)  #光标移到最后，这样就会自动显示出来
        QtWidgets.QApplication.processEvents()  #一定加上这个功能，不然有卡顿

    def messageDialog(self, mess_str):
        msg_box = QMessageBox(QMessageBox.Warning, 'Warning', mess_str)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui_main()
    t1 = threading.Thread(target=window.show())
    t1.start()
    sys.exit(app.exec_())