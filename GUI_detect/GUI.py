# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# import pyqtgraph as pg
from PyQt5 import QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, Monitor):

        if not Monitor.objectName():
            Monitor.setObjectName(u"Monitor")
        Monitor.setEnabled(True)
        Monitor.resize(1920, 1080)
        Monitor.setStyleSheet("#Monitor{border-image:url(./background_logo/background.jpg);}")
        Monitor.setWindowIcon(QIcon("./background_logo/logo.jpg"))
        
        self.centralwidget = QWidget(Monitor)
        self.centralwidget.setObjectName(u"centralwidget")
        self.listWidget = QListWidget(self.centralwidget)
        self.listWidget.setObjectName(u"listWidget")
        self.listWidget.setGeometry(QRect(20, 70, 331, 871))
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(30, 90, 301, 121))
        self.checkBox_5 = QCheckBox(self.groupBox)
        self.checkBox_5.setObjectName(u"checkBox_5")
        self.checkBox_5.setGeometry(QRect(20, 90, 101, 19))
        self.layoutWidget = QWidget(self.groupBox)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(20, 30, 291, 21))
        self.horizontalLayout_3 = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.checkBox = QCheckBox(self.layoutWidget)
        self.checkBox.setObjectName(u"checkBox")

        self.horizontalLayout_3.addWidget(self.checkBox)

        self.checkBox_2 = QCheckBox(self.layoutWidget)
        self.checkBox_2.setObjectName(u"checkBox_2")

        self.horizontalLayout_3.addWidget(self.checkBox_2)

        self.layoutWidget1 = QWidget(self.groupBox)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(20, 60, 291, 21))
        self.horizontalLayout_4 = QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.checkBox_3 = QCheckBox(self.layoutWidget1)
        self.checkBox_3.setObjectName(u"checkBox_3")

        self.horizontalLayout_4.addWidget(self.checkBox_3)

        self.checkBox_4 = QCheckBox(self.layoutWidget1)
        self.checkBox_4.setObjectName(u"checkBox_4")

        self.horizontalLayout_4.addWidget(self.checkBox_4)

        self.pushButton_3 = QPushButton(self.groupBox)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(170, 90, 81, 21))
        

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(30, 230, 301, 351))
        self.groupBox_3 = QGroupBox(self.groupBox_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(20, 30, 261, 71))
        self.label = QLabel(self.groupBox_3)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(30, 31, 60, 16))
        self.pushButton_4 = QPushButton(self.groupBox_3)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(160, 27, 81, 21))
        

        self.groupBox_4 = QGroupBox(self.groupBox_2)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(20, 110, 261, 71))
        self.pushButton_6 = QPushButton(self.groupBox_4)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(160, 30, 81, 21))
        

        self.label_2 = QLabel(self.groupBox_4)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(30, 31, 60, 16))
        self.groupBox_5 = QGroupBox(self.groupBox_2)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(20, 190, 261, 151))
        self.textEdit = QLineEdit(self.groupBox_5)
        self.textEdit.setObjectName(u"LineEdit")
        self.textEdit.setGeometry(QRect(10, 20, 241, 91))

        self.pushButton = QPushButton(self.groupBox_5)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(20, 120, 101, 21))
        font = QFont()
        font.setFamily(u"Arial")
        font.setPointSize(8)
        self.pushButton.setFont(font)
        

        self.pushButton_2 = QPushButton(self.groupBox_5)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(140, 120, 101, 21))
        self.pushButton_2.setFont(font)
        

        self.groupBox_6 = QGroupBox(self.centralwidget)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setGeometry(QRect(380, 70, 1251, 871))
        self.groupBox_6.setStyleSheet('QGroupBox {color: white;}')
        self.label_3 = QLabel(self.groupBox_6)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 25, 1231, 840))
        font1 = QFont()
        font1.setFamily(u"Arial")
        font1.setPointSize(14)
        self.label_3.setFont(font1)
        self.groupBox_7 = QGroupBox(self.centralwidget)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setGeometry(QRect(40, 590, 291, 341))

        # 图表显示
        self.label_4 = QLabel(self.groupBox_7)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(20, 30, 261, 221))

        
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(580, 20, 621, 31))
        self.label_5.setStyleSheet('color:white;')
        font2 = QFont()
        font2.setFamily(u"Arial")
        font2.setPointSize(18)
        self.label_5.setFont(font2)
        Monitor.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(Monitor)
        self.statusbar.setObjectName(u"statusbar")
        Monitor.setStatusBar(self.statusbar)

        # 开始检测
        self.pushButton_5 = QPushButton(self.centralwidget)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(1640, 76, 251, 71))
        

        # 状态信息
        self.groupBox_8 = QGroupBox(self.centralwidget)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setGeometry(QRect(1640, 350, 251, 591))
        self.groupBox_8.setAutoFillBackground(True)
        self.textBrowser = QTextBrowser(self.groupBox_8)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(10, 20, 231, 561))
        self.verticalScrollBar = QScrollBar(self.groupBox_8)
        self.verticalScrollBar.setObjectName(u"verticalScrollBar")
        self.verticalScrollBar.setGeometry(QRect(220, 10, 20, 569))
        self.verticalScrollBar.setOrientation(Qt.Vertical)
        self.pushButton_7 = QPushButton(self.centralwidget)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(1640, 170, 251, 71))
        self.pushButton_8 = QPushButton(self.centralwidget)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setGeometry(QRect(1640, 260, 251, 71))
        self.retranslateUi(Monitor)

        QMetaObject.connectSlotsByName(Monitor)
    # setupUi

    def retranslateUi(self, Monitor):
        Monitor.setWindowTitle(QCoreApplication.translate("Monitor", u"Monitor", None))
        self.groupBox.setTitle(QCoreApplication.translate("Monitor", u"\u68c0\u6d4b\u76ee\u6807", None))
        self.checkBox_5.setText(QCoreApplication.translate("Monitor", u"\u6469\u6258\u8f66", None))
        self.checkBox.setText(QCoreApplication.translate("Monitor", u"\u884c\u4eba", None))
        self.checkBox_2.setText(QCoreApplication.translate("Monitor", u"\u6c7d\u8f66", None))
        self.checkBox_3.setText(QCoreApplication.translate("Monitor", u"\u516c\u4ea4\u8f66", None))
        self.checkBox_4.setText(QCoreApplication.translate("Monitor", u"\u81ea\u884c\u8f66", None))
        self.pushButton_3.setText(QCoreApplication.translate("Monitor", u"\u786e\u5b9a", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Monitor", u"\u68c0\u6d4b\u7c7b\u578b", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Monitor", u"\u56fe\u7247\u68c0\u6d4b", None))
        self.label.setText(QCoreApplication.translate("Monitor", u"\u56fe\u7247\u8def\u5f84", None))
        self.pushButton_4.setText(QCoreApplication.translate("Monitor", u"\u4e0a\u4f20\u6587\u4ef6", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Monitor", u"\u6587\u4ef6\u89c6\u9891\u68c0\u6d4b", None))
        self.pushButton_6.setText(QCoreApplication.translate("Monitor", u"\u4e0a\u4f20\u6587\u4ef6", None))
        self.label_2.setText(QCoreApplication.translate("Monitor", u"\u89c6\u9891\u8def\u5f84", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Monitor", u"rtsp\u89c6\u9891\u6d41\u68c0\u6d4b", None))
        self.pushButton.setText(QCoreApplication.translate("Monitor", u"\u4e0a\u4f20rtsp\u5730\u5740", None))
        self.pushButton_2.setText(QCoreApplication.translate("Monitor", u"\u53d6\u6d88", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("Monitor", u"\u76d1\u63a7\u753b\u9762", None))
        # self.groupBox_7.setTitle("图表显示")
        self.label_5.setText(QCoreApplication.translate("Monitor", u"                               \u5b9e\u65f6\u76d1\u63a7\u7cfb\u7edfV1.0", None))
        self.pushButton_5.setText("加载模型")
        self.groupBox_8.setTitle(QCoreApplication.translate("Monitor", u"\u72b6\u6001\u4fe1\u606f", None))
        self.pushButton_7.setText("开始检测")
        self.pushButton_8.setText("清除画面")
    # retranslateUi

    