from typing import Callable, List, Any, Tuple


class ASLSignal:
    def __init__(self, *types: Any) -> None:
        self._types: Tuple[type, ...] = types
        self._connections: List[Callable[..., None]] = []

    def connect(self, func: Callable[..., None]) -> None:
        if callable(func):
            self._connections.append(func)
        else:
            raise TypeError(f"Function {func} is not callable")

    def emit(self, *args: Any) -> None:
        if len(args) != len(self._types):
            raise TypeError(f"Expected {len(self._types)} arguments, got {len(args)}")

        for arg, expected_type in zip(args, self._types):
            if not isinstance(arg, expected_type):
                raise TypeError(f"Argument {arg} is not of type {expected_type}")

        for func in self._connections:
            func(*args)
