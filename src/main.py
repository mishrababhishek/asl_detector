from PyQt5 import QtWidgets
from main_window import Ui_MainWindow
from ui_handler import UIHandler
import sys


def startApplication() -> None:
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    UIHandler(ui)
    sys.exit(app.exec_())


if __name__ == "__main__":
    startApplication()
