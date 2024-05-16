from src.main_window import Ui_MainWindow
from src.sign_language_detector import SignLanguageDetector
from src.image_processor import ImageProcessor
from src.models import ImageSignalArg
from src.locked_detector import LockedDetector
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt, QObject
import traceback
import asyncio
import pyttsx3


class UIHandler(QObject):
    def __init__(self, ui: Ui_MainWindow):
        super().__init__()
        try:
            self.tts_on = True
            self.speak_engine = pyttsx3.init()
            self.speak_engine.setProperty("volume", "1.0")
            self.ui = ui
            self.sign_language_detector = SignLanguageDetector()
            self.image_processor = ImageProcessor()
            self.locked_detector = LockedDetector(confidence=20)
            self.locked_detector.on_confidence_generated_signal.connect(
                self.onConfidenceReceive
            )
            self.image_processor.image_signal.connect(self.onFrameReceive)
            self.ui.startStopCameraButton.clicked.connect(self.toggleStartStop)
            self.ui.showFullCameraButton.clicked.connect(self.toggleCamera)
            self.ui.clearDetectionButton.clicked.connect(self.clearLockedDetection)
            self.ui.ttsButton.clicked.connect(self.toggleTTs)
            self.image_processor.start()
        except Exception as e:
            print(f"Error initializing UIHandler: {e}")
            traceback.print_exc()

    def toggleTTs(self) -> None:
        compare = self.ui.ttsButton.text() == "TTS On"
        self.tts_on = not self.tts_on
        self.ui.ttsButton.setText("TTS Off" if compare else "TTS On")

    def clearLockedDetection(self) -> None:
        self.ui.lockedDetectionLabel.setText("")

    async def speakAsync(self) -> None:
        self.speak_engine.stop()
        self.speak_engine.say(self.ui.lockedDetectionLabel.text())
        self.speak_engine.runAndWait()

    def onConfidenceReceive(self, char: str):
        self.ui.lockedDetectionLabel.setText(
            f"{self.ui.lockedDetectionLabel.text()}{char}"
        )
        if self.tts_on:
            asyncio.run(self.speakAsync())

    def toggleStartStop(self) -> None:
        compare = self.ui.startStopCameraButton.text() == "Start Camera"
        self.image_processor.start_stop_signal.emit()
        self.ui.startStopCameraButton.setText(
            "Stop Camera" if compare else "Start Camera"
        )

    def toggleCamera(self) -> None:
        compare = self.ui.showFullCameraButton.text() == "Show Full Camera"
        self.image_processor.show_only_hand_signal.emit()
        self.ui.showFullCameraButton.setText(
            "Show Hand Only" if compare else "Show Full Camera"
        )

    def onFrameReceive(self, arg: ImageSignalArg):
        try:

            height, width, _ = arg.display_image.shape
            bytes_per_line = 3 * width
            qt_img = QImage(
                arg.display_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_BGR888,
            )

            label_width = self.ui.videoLabel.width()
            label_height = self.ui.videoLabel.height()

            scaled_img = qt_img.scaled(label_width, label_height, Qt.KeepAspectRatio)

            centered_pixmap = QPixmap(label_width, label_height)
            centered_pixmap.fill(Qt.transparent)

            x = (label_width - scaled_img.width()) // 2
            y = (label_height - scaled_img.height()) // 2

            painter = QPainter(centered_pixmap)
            painter.drawPixmap(x, y, QPixmap.fromImage(scaled_img))
            painter.end()
            if arg.hand_image is not None:
                char = self.sign_language_detector.classify(arg.hand_image)
                self.ui.realtimeDetectionLabel.setText(char)
                self.locked_detector.update(char)
            self.ui.videoLabel.setPixmap(centered_pixmap)
        except RuntimeError as re:
            print(f"RuntimeError: {re}")
            traceback.print_exc()
        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
