import cv2
import json
import math
import torch
import pyttsx3
import threading
import asyncio
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from cvzone.HandTrackingModule import HandDetector
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class SignLanguageClassifer:
    def __init__(self, settings) -> None:
        self._processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
        self._model = AutoModelForImageClassification.from_pretrained(
            "RavenOnur/Sign-Language"
        )
        self._device = torch.device("cpu")
        if settings["use_cuda_if_available"]:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def classify(self, cv_img):
        inputs = self._processor(cv_img, return_tensors="pt").to(self._device)
        logits = self._model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        return self._model.config.id2label[predicted_label]


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_realtime_text_signal = pyqtSignal(str)
    change_confident_text_signal = pyqtSignal(str)

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self._run_flag = True
        self._classifier = SignLanguageClassifer(settings)
        self.saved_text_list = []
        self.confident_text: str = None

    def compare_list(self, text_list):
        return all(item == text_list[0] for item in text_list)

    def generate_confident_text(self, text):
        if len(self.saved_text_list) < self.settings["threshold"]:
            self.saved_text_list.insert(0, text)
        else:
            if self.compare_list(self.saved_text_list):
                if self.confident_text is None:
                    self.confident_text = self.saved_text_list[0]
                else:
                    self.confident_text += self.saved_text_list[0]
                self.saved_text_list = []
                self.saved_text_list.insert(0, text)
                self.change_confident_text_signal.emit(self.confident_text)
            else:
                self.saved_text_list.pop()
                self.saved_text_list.insert(0, text)

    def process_frame(self, x, y, w, h, cv_img, offset=20):
        img_white = np.ones((300, 300, 3), np.uint8) * 255
        croped_image = cv_img[y - offset : y + h + offset, x - offset : x + w + offset]
        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = 300 / h
            wCal = math.ceil(k * w)
            image_resize = cv2.resize(croped_image, (wCal, 300))
            wGap = math.ceil((300 - wCal) / 2)
            img_white[:, wGap : wCal + wGap] = image_resize
        else:
            k = 300 / w
            hCal = math.ceil(k * h)
            image_resize = cv2.resize(croped_image, (300, hCal))
            hGap = math.ceil((300 - hCal) / 2)
            img_white[hGap : hCal + hGap, :] = image_resize
        return img_white

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        hand_detector = HandDetector(maxHands=1)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                if self.settings["flip_image"]:
                    cv_img = cv2.flip(cv_img, 1)
                hands, cv_img = hand_detector.findHands(
                    cv_img, flipType=not self.settings["flip_image"]
                )
                if hands:
                    x, y, w, h = hands[0]["bbox"]
                    try:
                        hand_image = self.process_frame(x, y, w, h, cv_img)
                        output = self._classifier.classify(hand_image)
                        self.change_realtime_text_signal.emit(output)
                        self.generate_confident_text(output)
                    except:
                        pass
                else:
                    self.change_realtime_text_signal.emit("None")
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        self.change_realtime_text_signal.emit("None")
        self._run_flag = False
        self.wait()


class Ui_MainWindow(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.settings = json.load(open("./settings.json", "r"))
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("volume", 1.0)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 400)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(800, 400))
        MainWindow.setMaximumSize(QtCore.QSize(800, 400))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212, 128))
        brush.setStyle(Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212, 128))
        brush.setStyle(Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(30, 30, 30))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(212, 212, 212, 128))
        brush.setStyle(Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        MainWindow.setPalette(palette)
        MainWindow.setStyleSheet(
            "* {\n"
            "    background-color: #1E1E1E;\n"
            "    color: #D4D4D4;\n"
            "}\n"
            "\n"
            "QLabel{\n"
            "    max-height: 15\n"
            "}\n"
            "\n"
            "#camera_container,  #controls_container {\n"
            "    border: 1px solid #ffffff;\n"
            "}\n"
            "\n"
            "QPushButton, QComboBox, QLineEdit, QPlainTextEdit, QListView, QTreeView, QTabWidget {\n"
            "    background-color: #333333;\n"
            "    border: 1px solid #444444;\n"
            "    padding: 5px;\n"
            "    border-radius: 3px;\n"
            "}\n"
            "\n"
            "QPushButton:hover, QComboBox:hover, QLineEdit:hover, QPlainTextEdit:hover, QListView:hover, QTreeView:hover, QTabWidget:hover {\n"
            "    background-color: #444444;\n"
            "}\n"
            "\n"
            "QPushButton:pressed, QComboBox:pressed, QLineEdit:pressed, QPlainTextEdit:pressed, QListView:pressed, QTreeView:pressed, QTabWidget:pressed {\n"
            "    background-color: #555555;\n"
            "}\n"
            "\n"
            "QMenuBar, QMenu, QMenuBar::item, QMenu::item {\n"
            "    background-color: #2D2D30;\n"
            "    color: #D4D4D4;\n"
            "}\n"
            "\n"
            "QMenu::item:selected {\n"
            "    background-color: #555555;\n"
            "}\n"
            "\n"
            "QScrollBar:vertical {\n"
            "    border: 1px solid #444444;\n"
            "    background: #333333;\n"
            "    width: 10px;\n"
            "    margin: 0px 0px 0px 0px;\n"
            "}\n"
            "\n"
            "QScrollBar::handle:vertical {\n"
            "    background: #555555;\n"
            "    min-height: 20px;\n"
            "}\n"
            "\n"
            "QScrollBar::handle:vertical:hover {\n"
            "    background: #666666;\n"
            "}\n"
            "\n"
            "QScrollBar::handle:vertical:pressed {\n"
            "    background: #777777;\n"
            "}\n"
            ""
        )
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.camera_container = QtWidgets.QWidget(self.centralwidget)
        self.camera_container.setMinimumSize(QtCore.QSize(500, 0))
        self.camera_container.setMaximumSize(QtCore.QSize(500, 400))
        self.camera_container.setObjectName("camera_container")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.camera_container)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.video_label = QtWidgets.QLabel(self.camera_container)
        self.video_label.setMinimumSize(QtCore.QSize(470, 355))
        self.video_label.setMaximumSize(QtCore.QSize(16777215, 378))
        self.video_label.setObjectName("video_label")
        self.verticalLayout_5.addWidget(self.video_label, 1, Qt.AlignTop)
        self.horizontalLayout.addWidget(self.camera_container)
        self.video_thread = VideoThread(self.settings)
        self.video_thread.change_pixmap_signal.connect(self.update_video_label)
        self.video_thread.change_realtime_text_signal.connect(
            self.update_realtime_label
        )
        self.video_thread.change_confident_text_signal.connect(
            self.update_confident_label
        )
        self.controls_container = QtWidgets.QWidget(self.centralwidget)
        self.controls_container.setObjectName("controls_container")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.controls_container)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(self.controls_container)
        self.widget.setMaximumSize(QtCore.QSize(16777215, 70))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setMinimumSize(QtCore.QSize(0, 0))
        self.label.setMaximumSize(QtCore.QSize(16777215, 15))
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.realtime_detection_label = QtWidgets.QLabel(self.widget)
        self.realtime_detection_label.setObjectName("realtime_detection_label")
        self.verticalLayout_2.addWidget(self.realtime_detection_label)
        self.verticalLayout.addWidget(self.widget, 0, Qt.AlignTop)
        self.widget_2 = QtWidgets.QWidget(self.controls_container)
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 70))
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget_2)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.confident_detection_label = QtWidgets.QLabel(self.widget_2)
        self.confident_detection_label.setObjectName("confident_detection_label")
        self.verticalLayout_3.addWidget(self.confident_detection_label)
        self.verticalLayout.addWidget(self.widget_2, 0, Qt.AlignTop)
        self.widget_3 = QtWidgets.QWidget(self.controls_container)
        self.widget_3.setMaximumSize(QtCore.QSize(16777215, 100))
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.widget_3)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_4.addWidget(self.label_5)
        self.radioButton = QtWidgets.QRadioButton(self.widget_3)
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout_4.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.widget_3)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout_4.addWidget(self.radioButton_2)
        self.verticalLayout.addWidget(self.widget_3)
        spacerItem = QtWidgets.QSpacerItem(
            20, 100, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout.addItem(spacerItem)
        self.clear_confident_detection = QtWidgets.QPushButton(self.controls_container)
        self.clear_confident_detection.setObjectName("clear_confident_detection")
        self.clear_confident_detection.clicked.connect(self.clear_confident_text)
        self.verticalLayout.addWidget(self.clear_confident_detection)
        self.start_stop_camera = QtWidgets.QPushButton(self.controls_container)
        self.start_stop_camera.setObjectName("start_stop_camera")
        self.start_stop_camera.clicked.connect(self.toggle_camera)
        self.verticalLayout.addWidget(self.start_stop_camera)
        self.horizontalLayout.addWidget(self.controls_container, 0, Qt.AlignTop)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.closeEvent = self.close_event
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def clear_confident_text(self):
        _translate = QtCore.QCoreApplication.translate
        self.confident_detection_label.setText(_translate("MainWindow", "None"))
        self.video_thread.confident_text = None

    def toggle_camera(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.start_stop_camera.setText("Start Camera")
        else:
            self.video_thread._run_flag = True
            self.video_thread.start()
            self.start_stop_camera.setText("Stop Camera")

    def update_video_label(self, cv_img):
        height, width, _ = cv_img.shape
        bytes_per_line = 3 * width
        qt_img = QImage(
            cv_img.data, width, height, bytes_per_line, QImage.Format_BGR888
        )
        qt_pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(qt_pixmap)

    def update_realtime_label(self, output):
        _translate = QtCore.QCoreApplication.translate
        self.realtime_detection_label.setText(_translate("MainWindow", output))

    async def speak_async(self, text):
        if self.radioButton.isChecked():
            self.tts_engine.stop()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def speak(self, text):
        asyncio.run(self.speak_async(text))

    def update_confident_label(self, output):
        _translate = QtCore.QCoreApplication.translate
        self.confident_detection_label.setText(_translate("MainWindow", output))
        threading.Thread(target=self.speak, args=(output,)).start()

    def close_event(self, event):
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.start_stop_camera.setText("Start Camera")
        super().customEvent(event)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate("MainWindow", "American Sign Language Detector")
        )
        self.label.setText(_translate("MainWindow", "RealTime Detection"))
        self.realtime_detection_label.setText(_translate("MainWindow", "None"))
        self.label_3.setText(_translate("MainWindow", "Confident Detection"))
        self.confident_detection_label.setText(_translate("MainWindow", "None"))
        self.label_5.setText(_translate("MainWindow", "Text to Speech Mode"))
        self.radioButton.setText(_translate("MainWindow", "Enabled"))
        self.radioButton_2.setText(_translate("MainWindow", "Disabled"))
        self.clear_confident_detection.setText(_translate("MainWindow", "Clear"))
        self.start_stop_camera.setText(_translate("MainWindow", "Start Camera"))
