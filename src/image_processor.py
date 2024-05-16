from cvzone.HandTrackingModule import HandDetector
from cv2 import VideoCapture, flip, destroyAllWindows, resize, cvtColor, COLOR_BGR2RGB
from PyQt5.QtCore import QThread
from models import ImageSignalArg
from asl_signal import ASLSignal
import traceback
import numpy as np


class ImageProcessor(QThread):
    def __init__(self) -> None:
        super().__init__()
        self.image_signal = ASLSignal(ImageSignalArg)
        self.start_stop_signal = ASLSignal()
        self.show_only_hand_signal = ASLSignal()
        self._start_stop = False
        self._show_only_hand = False
        self.start_stop_signal.connect(self.toggle_start_stop)
        self.show_only_hand_signal.connect(self.toggle_show_only_hand)

    def toggle_start_stop(self):
        self._start_stop = not self._start_stop

    def toggle_show_only_hand(self):
        self._show_only_hand = not self._show_only_hand

    @staticmethod
    def extract_and_resize_hand(cv2_image, hand):
        x, y, w, h = hand["bbox"]
        x -= 20
        y -= 20
        w += 40
        h += 40
        hand_image = cv2_image[
            max(0, y) : min(y + h, cv2_image.shape[0]),
            max(0, x) : min(x + w, cv2_image.shape[1]),
        ]
        hand_aspect_ratio = w / h
        canvas_size = 300
        if hand_aspect_ratio > 1:
            new_width = canvas_size
            new_height = int(canvas_size / hand_aspect_ratio)
        else:
            new_width = int(canvas_size * hand_aspect_ratio)
            new_height = canvas_size
        resized_hand_image = resize(hand_image, (new_width, new_height))
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        x_offset = (canvas_size - new_width) // 2
        y_offset = (canvas_size - new_height) // 2
        canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
            resized_hand_image
        )
        return canvas

    def run(self) -> None:
        try:
            video_capture = VideoCapture(0)
            hand_detector = HandDetector(maxHands=1)
            while True:
                if not self._start_stop:
                    continue
                try:
                    ret, cv2_image = video_capture.read()
                    if ret:
                        cv2_image = flip(cv2_image, 1)
                        hands, cv2_image_with_hand = hand_detector.findHands(
                            cv2_image, flipType=False
                        )
                        if hands:
                            only_hand = self.extract_and_resize_hand(
                                cv2_image_with_hand, hands[0]
                            )
                            if self._show_only_hand:
                                self.image_signal.emit(
                                    ImageSignalArg(only_hand, only_hand)
                                )
                            else:
                                self.image_signal.emit(
                                    ImageSignalArg(cv2_image_with_hand, only_hand)
                                )
                        else:
                            if not self._show_only_hand:
                                self.image_signal.emit(ImageSignalArg(cv2_image, None))
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    traceback.print_exc()
        except Exception as e:
            print(f"Error initializing video capture or hand detector: {e}")
            traceback.print_exc()
        finally:
            video_capture.release()
            destroyAllWindows()
