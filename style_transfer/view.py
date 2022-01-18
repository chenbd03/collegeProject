# -*- coding: utf-8 -*-
"""
@Author billie, Belton
@Date 2020/11/23 23:38
@Describe 
"""

import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QPushButton
from PyQt5.QtGui import QIcon

from neural_style import neural_style

import sys
import os

class StyleMove(QWidget):
    def __init__(self):
        super(StyleMove,self).__init__()
        self.initUI() # 使用initUI()方法创建一个GUI
        self.picPath = "/"

    def initUI(self):

        # 窗体初始化
        self.setWindowTitle('F.G.Q.Y')
        self.setWindowIcon(QIcon('./example_img/logo.png'))
        self.setGeometry(0, 0, 800, 600) #x,y,宽，高；resize()和move()的结合体

        # 各个盒子初始化
        self.createExampleBox() # 展示区
        self.createOperateBox() # 交互区
        self.createShowResultBox()  # 显示结果区

        # 主体布局
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.exampleBox,0,0,8,1)
        mainLayout.addWidget(self.operateBox,0,1,1,5)
        mainLayout.addWidget(self.showResultBox,1,1,7,5)

        self.setLayout(mainLayout)
        # self.setWindowOpacity(0.8)  # 设置窗口透明度
        # 背景 颜色
        palette1 = QPalette()
        palette1.setColor(palette1.Background, QColor(193,210,240))
        self.setPalette(palette1)

    # 盒子 - 示例图展示
    def createExampleBox(self):
        examp_1 = QLabel("示例图1") # 四张示例图
        examp_2 = QLabel("示例图2")
        examp_3 = QLabel("示例图3")
        examp_4 = QLabel("示例图4")

        label_1 = QLabel("风格1") # 四张示例图的说明文字，可修改为风格名称
        label_2 = QLabel("风格2")
        label_3 = QLabel("风格3")
        label_4 = QLabel("风格4")

        label_1.setAlignment(Qt.AlignCenter) # 居中对齐
        label_2.setAlignment(Qt.AlignCenter)
        label_3.setAlignment(Qt.AlignCenter)
        label_4.setAlignment(Qt.AlignCenter)

        examp_1.setPixmap(QPixmap('./example_img/1.png')) # 添加图片
        examp_2.setPixmap(QPixmap('./example_img/2.png'))
        examp_3.setPixmap(QPixmap('./example_img/3.png'))
        examp_4.setPixmap(QPixmap('./example_img/4.png'))

        layout = QGridLayout() # 创建布局，布局方式为网格布局 - QGridLayout
        layout.addWidget(examp_1,0,0) # 添加控件到布局
        layout.addWidget(label_1,1,0)
        layout.addWidget(examp_2,2,0)
        layout.addWidget(label_2,3,0)
        layout.addWidget(examp_3,4,0)
        layout.addWidget(label_3,5,0)
        layout.addWidget(examp_4,6,0)
        layout.addWidget(label_4,7,0)

        self.exampleBox = QGroupBox("Example Box") # 创建盒子，命名为"Example layout"
        self.exampleBox.setLayout(layout) # 添加布局到盒子


    # 盒子 - 交互操作
    def createOperateBox(self):

        btnGetPic = QPushButton("选择图片") # 按钮 - 获取图片
        btnGetPic.clicked.connect(self.loadPic) # 设置关联函数
        btnGetPic.setStyleSheet("QPushButton{color:black}"
                                       "QPushButton:hover{color:lightgreen;}"
                                       "QPushButton{background-color:white}"
                                       "QPushButton{border:2px}"
                                       "QPushButton{border-radius:10px}"
                                       "QPushButton{padding:8px 4px}")

        btnTransform = QPushButton("生成图片")  # 按钮 - 制作图片
        btnTransform.clicked.connect(self.transformPic)
        btnTransform.setStyleSheet("QPushButton{color:none}"
                                       "QPushButton:hover{color:lightgreen;}"
                                       "QPushButton{background-color:white}"
                                       "QPushButton{border:2px}"
                                       "QPushButton{border-radius:10px}"
                                       "QPushButton{padding:8px 4px}")

        btnSaveImg = QPushButton("保存图片")    # 按钮 - 保存图片
        btnSaveImg.clicked.connect(self.saveImg)
        btnSaveImg.setStyleSheet("QPushButton{color:none}"
                                       "QPushButton:hover{color:lightgreen;}"
                                       "QPushButton{background-color:white}"
                                       "QPushButton{border:2px}"
                                       "QPushButton{border-radius:10px}"
                                       "QPushButton{padding:8px 4px}")

        self.style = ""  # 初始风格
        combo = QComboBox()  # 下拉列表 - 选择风格
        combo.addItem("请选择想要的风格")
        combo.addItem("风格1")
        combo.addItem("风格2")
        combo.addItem("风格3")
        combo.addItem("风格4")
        combo.activated[str].connect(self.onActivated)  # 设置关联函数
        combo.setStyleSheet(
            "font-size:14px;font-weight:500;border:2px;border-radius:10px;padding:8px 4px;background-color : none;hover{color:lightgreen}")

        layout = QHBoxLayout()  # 创建布局，布局方式为水平布局 - QHBoxLayout
        layout.addWidget(btnGetPic) # 添加控件到布局
        layout.addWidget(btnTransform)
        layout.addWidget(btnSaveImg)
        layout.addWidget(combo)

        self.operateBox = QGroupBox("Operation Box") # 创建盒子
        self.operateBox.setLayout(layout) # 添加layout到盒子

    # 盒子 - 原图与制作的结果展示
    def createShowResultBox(self):
        self.openPic = QLabel() # 打开的图片
        self.openPic.setFixedSize(500,300)

        self.resultPic = QLabel() # 生成的图片
        self.resultPic.setFixedSize(500,300)

        layout = QGridLayout()  # 创建组件，布局方式为：QGridLayout
        layout.addWidget(self.openPic)
        layout.addWidget(self.resultPic)

        self.showResultBox = QGroupBox("Show Box")  # 创建盒子
        self.showResultBox.setLayout(layout)

    # 事件 - 获取下拉框的选择
    def onActivated(self,style):
        self.style = style

    # 事件 - 打开本地文件
    def loadPic(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '选择图片', self.picPath, 'Image files(*.jpg *.gif *.png)')
        self.picPath = '/'.join(self.fname.split('/')[:-1])
        print(self.picPath)

        self.openPic.setPixmap(QPixmap(self.fname).scaled(500, 300))
        # self.resultPic.setPixmap(None)
        print(self.fname)
    # 事件 - 保存图片
    def saveImg(self):
        file_path = QFileDialog.getSaveFileName(self, "save file", self.picPath, 'Image files(*.jpg *.gif *.png)')

        # 事件 - 转换图片风格
    def transformPic(self):
        if self.style == "请选择想要的风格" or self.style == "":
            return self.resultPic.setText("请选择风格!!!")
        elif self.style == "风格1":
            model = "./saved_models/mosaic.pth"
        elif self.style == "风格2":
            model = "./saved_models/rain_princess.pth"
        elif self.style == "风格3":
            model = "./saved_models/udnie.pth"
        elif self.style == "风格4":
            model = "./saved_models/candy.pth"


        img_name = str(self.fname).split("/")[-1]
        out_img_path = os.getcwd() + os.sep + "images" + os.sep + "output" + os.sep + img_name
        self.resultPic.setText("图片正在生成,请稍后...")
        PyQt5.QtWidgets.QApplication.processEvents()
        PyQt5.QtWidgets.QApplication.processEvents()
        self.refname = neural_style.stylize(model, self.fname, out_img_path)
        self.resultPic.setPixmap(QPixmap(self.refname).scaled(500, 300))


if __name__ == '__main__':
    app = QApplication(sys.argv) # 创建一个应用对象；sys.argv参数是一个来自命令行的参数列表

    # 1. 构造简单界面
    # w = QWidget()

    # 2. 实例化界面类
    w = StyleMove()
    w.show()

    sys.exit(app.exec_())