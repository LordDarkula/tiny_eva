from typing import Callable, Any
from os import PathLike

import torch

from tiny_eva.decoder import Frame


class UDF:
    def __init__(self, func: Callable) -> None:
        self.func: Callable = func

    @classmethod
    def from_callable(cls, func: Callable):
        return cls(func)

    @classmethod
    def from_torch_hub(cls, path: PathLike, name: str, pretrained: bool = True):
        def model_func(frame: Frame):
            model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=pretrained
            )
            return model(frame.source)

        return cls(model_func)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
