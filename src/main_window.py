# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'designer.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(897, 380)
        MainWindow.setStyleSheet(
            "\n" "    background-color: #0f0f0f;\n" "    color: #ffffff;\n" "   "
        )
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.videoLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(7)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.videoLabel.sizePolicy().hasHeightForWidth())
        self.videoLabel.setSizePolicy(sizePolicy)
        self.videoLabel.setMinimumSize(QtCore.QSize(640, 360))
        self.videoLabel.setStyleSheet(
            "\n"
            "          background-color: #1c1c1c;\n"
            "          border-radius: 10px;\n"
            "          padding: 10px;\n"
            "         "
        )
        self.videoLabel.setText("")
        self.videoLabel.setObjectName("videoLabel")
        self.horizontalLayout_4.addWidget(self.videoLabel)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("\n" "            color: #ff9900;\n" "           ")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.realtimeDetectionLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.realtimeDetectionLabel.sizePolicy().hasHeightForWidth()
        )
        self.realtimeDetectionLabel.setSizePolicy(sizePolicy)
        self.realtimeDetectionLabel.setStyleSheet(
            "\n"
            "            background-color: #1c1c1c;\n"
            "            border-radius: 10px;\n"
            "            padding: 10px;\n"
            "           "
        )
        self.realtimeDetectionLabel.setText("")
        self.realtimeDetectionLabel.setObjectName("realtimeDetectionLabel")
        self.verticalLayout.addWidget(self.realtimeDetectionLabel)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("\n" "            color: #ff9900;\n" "           ")
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.lockedDetectionLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.lockedDetectionLabel.sizePolicy().hasHeightForWidth()
        )
        self.lockedDetectionLabel.setSizePolicy(sizePolicy)
        self.lockedDetectionLabel.setStyleSheet(
            "\n"
            "            background-color: #1c1c1c;\n"
            "            border-radius: 10px;\n"
            "            padding: 10px;\n"
            "           "
        )
        self.lockedDetectionLabel.setText("")
        self.lockedDetectionLabel.setObjectName("lockedDetectionLabel")
        self.verticalLayout.addWidget(self.lockedDetectionLabel)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.showFullCameraButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.showFullCameraButton.sizePolicy().hasHeightForWidth()
        )
        self.showFullCameraButton.setSizePolicy(sizePolicy)
        self.showFullCameraButton.setStyleSheet(
            "\n"
            "              background-color: #ff9900;\n"
            "              color: #ffffff;\n"
            "              border-radius: 10px;\n"
            "              padding: 10px;\n"
            "             "
        )
        self.showFullCameraButton.setCheckable(True)
        self.showFullCameraButton.setObjectName("showFullCameraButton")
        self.horizontalLayout_2.addWidget(self.showFullCameraButton)
        self.ttsButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ttsButton.sizePolicy().hasHeightForWidth())
        self.ttsButton.setSizePolicy(sizePolicy)
        self.ttsButton.setStyleSheet(
            "\n"
            "              background-color: #ff9900;\n"
            "              color: #ffffff;\n"
            "              border-radius: 10px;\n"
            "              padding: 10px;\n"
            "             "
        )
        self.ttsButton.setCheckable(True)
        self.ttsButton.setObjectName("ttsButton")
        self.horizontalLayout_2.addWidget(self.ttsButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.startStopCameraButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.startStopCameraButton.sizePolicy().hasHeightForWidth()
        )
        self.startStopCameraButton.setSizePolicy(sizePolicy)
        self.startStopCameraButton.setStyleSheet(
            "\n"
            "              background-color: #ff9900;\n"
            "              color: #ffffff;\n"
            "              border-radius: 10px;\n"
            "              padding: 10px;\n"
            "             "
        )
        self.startStopCameraButton.setObjectName("startStopCameraButton")
        self.horizontalLayout_3.addWidget(self.startStopCameraButton)
        self.clearDetectionButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.clearDetectionButton.sizePolicy().hasHeightForWidth()
        )
        self.clearDetectionButton.setSizePolicy(sizePolicy)
        self.clearDetectionButton.setStyleSheet(
            "\n"
            "              background-color: #ff9900;\n"
            "              color: #ffffff;\n"
            "              border-radius: 10px;\n"
            "              padding: 10px;\n"
            "             "
        )
        self.clearDetectionButton.setObjectName("clearDetectionButton")
        self.horizontalLayout_3.addWidget(self.clearDetectionButton)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.horizontalLayout_4.setStretch(0, 7)
        self.horizontalLayout_4.setStretch(1, 3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate("MainWindow", "American Sign Language Detection")
        )
        self.label.setText(_translate("MainWindow", "Realtime Detection"))
        self.label_3.setText(_translate("MainWindow", "Locked Detection"))
        self.showFullCameraButton.setText(_translate("MainWindow", "Show Hand Only"))
        self.ttsButton.setText(_translate("MainWindow", "TTS On"))
        self.startStopCameraButton.setText(_translate("MainWindow", "Start Camera"))
        self.clearDetectionButton.setText(
            _translate("MainWindow", "Clear Locked Detection")
        )
