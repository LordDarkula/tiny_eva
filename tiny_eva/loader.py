from typing import Any, Callable
from os import PathLike

import torch  # type: ignore

from tiny_eva.frame import Frame


class UDF:
    def __init__(self, func: Callable) -> None:
        self.func: Callable = func

    @classmethod
    def from_callable(cls, func: Callable):
        return cls(func)

    @classmethod
    def from_torch_hub(cls, path: PathLike, name: str, pretrained: bool = True):
        """
        Loads model from pytorch hub. Downloads model, so this will take time.

        Arguments:
            path: hub path to model repo
            name: name of hub model
            pretrained: should fetched model be pretrained
        """

        def model_func(frame: Frame):
            model = torch.hub.load(path, name, pretrained=pretrained)
            return model(frame.source).pandas().xyxy

        return cls(model_func)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
