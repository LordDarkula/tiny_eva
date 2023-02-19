from typing import Any, Callable

from tiny_eva.frame import GenericFrame
from tiny_eva.loaders.generic_loader import GenericLoader


class CallableLoader(GenericLoader):
    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(self, frame: GenericFrame, **kwds: Any) -> Any:
        return self._func(frame, **kwds)
