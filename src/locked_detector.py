from collections import deque
from src.asl_signal import ASLSignal


class LockedDetector:
    def __init__(self, confidence: int = 10) -> None:
        self._confidence = confidence
        self._preserved: deque[str] = deque()
        self.on_confidence_generated_signal = ASLSignal(str)

    def update(self, char: str) -> None:
        if len(self._preserved) < self._confidence:
            self._preserved.append(char)
        else:
            if self._check_all_same(self._preserved, char):
                self._preserved.clear()
                self.on_confidence_generated_signal.emit(char)
            else:
                self._preserved.popleft()
                self._preserved.append(char)

    def _check_all_same(self, arg: deque[str], compare: str) -> bool:
        if len(arg) <= 1:
            return True
        return all(element == compare for element in arg)
