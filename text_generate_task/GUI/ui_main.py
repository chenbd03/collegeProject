from .main_GUI import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow,QWidget
from textGen.poetry_api import LSTMPoetryModel,ModelConfig
from PyQt5.QtWidgets import QMessageBox

class Ui_main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Ui_main, self).__init__()
        # 初始化
        self.setupUi(self)
        # 模型加载
        self.model = LSTMPoetryModel(ModelConfig)
        print("模型加载成功..")
        # 按钮触发事件
        self.pushButton.clicked.connect(self.btn_1_function)
        self.pushButton_2.clicked.connect(self.btn_2_function)
        self.pushButton_3.clicked.connect(self.btn_3_function)
    
    def predict_text(self, text=None, is_random=False):
        """
        生成预测古诗的结果显示
        """
        if is_random:  # 如果是随机生成的诗

            text = self.model.predict_random()
        else:
            if len(text) == 4:   # 如果是藏头诗
                text = self.model.predict_hide(text)
            else:           # 否则是一个字的古诗
                text = self.model.predict_first(text)
        return text

    def btn_1_function(self):
        """
        随机生成古诗,再显示在文本框
        """
        text = self.predict_text(is_random=True)
        self.textBrowser.setText(text)

    def btn_2_function(self):
        """
        随机生成古诗,再显示在文本框
        """
        text = self.lineEdit.text()
        if len(text) != 4:
            self.messageDialog("输出的文本长度不是4个,请重新输入!")
            self.lineEdit.clear()
        else:
            result = self.predict_text(text=text)
            self.textBrowser.setText(result)
            self.lineEdit.clear()
        
    def btn_3_function(self):
        """
        随机生成古诗,再显示在文本框
        """
        text = self.lineEdit_2.text()
        if len(text) != 1:
            self.messageDialog("输出的文本长度不是1个,请重新输入!")
            self.lineEdit_2.clear()
        else:
            result = self.predict_text(text=text)
            self.textBrowser.setText(result)
            self.lineEdit_2.clear()
    
    def messageDialog(self, mess_str):
        """
        信息警告
        """
        msg_box = QMessageBox(QMessageBox.Warning, 'Warning', mess_str)
        msg_box.exec_()
    

    
