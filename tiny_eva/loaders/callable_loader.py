from typing import Callable, Any

from tiny_eva.loaders.generic_loader import GenericLoader
from tiny_eva.frame import AbstractFrame


class CallableLoader(GenericLoader):
    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(self, frame: AbstractFrame, **kwds: Any) -> Any:
        return self._func(frame, **kwds)
