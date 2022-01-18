# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'designerYJMxou.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1207, 833)
        MainWindow.setStyleSheet("#MainWindow{border-image:url(./GUI/background.jpg);}")  # 背景图片
        MainWindow.setWindowIcon(QIcon("./GUI/logo.jpg"))       # 图标图片
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        # 随机生成诗的按钮
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(140, 200, 161, 111))
        # 藏头诗按钮
        self.pushButton_2 = QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(510, 200, 161, 111))
        # 一个字古诗的按钮
        self.pushButton_3 = QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(890, 200, 161, 111))
        # 请输入字
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(460, 170, 72, 15))
        # 请输入
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(830, 170, 72, 15))
        # 输入框文本
        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(520, 160, 141, 31))
        # 输入框文本
        self.lineEdit_2 = QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        self.lineEdit_2.setGeometry(QRect(900, 160, 141, 31))
        # 显示生成古诗的文本
        self.textBrowser = QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName(u"textBrowser")
        self.textBrowser.setGeometry(QRect(200, 390, 821, 351))
        # 请输入
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(160, 360, 101, 16))

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(470, 60, 291, 61))
        self.label_4.setStyleSheet(u"font: 75 18pt \"Arial\";")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1207, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)


    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"\u968f\u673a\u751f\u6210\u4e00\u9996\u8bd7", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\u751f\u6210\u85cf\u5934\u8bd7\uff08\u56db\u4e2a\u5b57\uff09", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"\u7ed9\u51fa\u4e00\u4e2a\u5b57\u751f\u6210\u53e4\u8bd7", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u8bf7\u8f93\u5165\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u8bf7\u8f93\u5165\uff1a", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u53e4\u8bd7\u751f\u6210\u7ed3\u679c\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u4e94\u8a00\u5f8b\u8bd7\u751f\u6210\u5668V1.0", None))
    # retranslateUi

