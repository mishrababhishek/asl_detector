from PyQt5 import QtWidgets
from src.main_window import Ui_MainWindow
from src.ui_handler import UIHandler
import sys


def startApplication() -> None:
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    UIHandler(ui)
    sys.exit(app.exec_())
