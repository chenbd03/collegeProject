from GUI.ui_main import Ui_main
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import threading

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # w_main = QMainWindow()#创建一个主窗体（必须要有一个主窗体）
    window = Ui_main()
    t1 = threading.Thread(target=window.show())
    t1.start()
    sys.exit(app.exec_())