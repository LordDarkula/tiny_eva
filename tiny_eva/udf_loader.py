from typing import Any, Callable
from os import PathLike
from enum import Enum, auto
from abc import ABCMeta, abstractmethod

import torch  # type: ignore

from tiny_eva.frame import Frame


class ModelSource(Enum):
    CALLABLE = auto()
    TORCH_HUB = auto()


class AbstractLoader(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, frame: Frame, **kwds: Any) -> Any:
        """Run model on frame and return result."""


class CallableLoader(AbstractLoader):
    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(self, frame: Frame, **kwds: Any) -> Any:
        return self._func(frame, *kwds)


class TorchHubLoader(AbstractLoader):
    pass


class UDF:
    def __init__(self, func: Any, model_source: ModelSource) -> None:
        self.func: Any = func
        self.model_source: ModelSource = model_source

    @staticmethod
    def from_callable(func: Callable):
        return CallableLoader(func)

    @classmethod
    def from_torch_hub(cls, path: PathLike, name: str, pretrained: bool = True):
        """
        Loads model from pytorch hub. Downloads model, so this will take time.

        Arguments:
            path: hub path to model repo
            name: name of hub model
            pretrained: should fetched model be pretrained
        """
        model = torch.hub.load(path, name, pretrained=pretrained)
        return cls(model, ModelSource.TORCH_HUB)

    def __call__(self, frame: Frame, **kwds: Any) -> Any:
        if self.model_source == ModelSource.TORCH_HUB:
            return self.func(frame.source).pandas().xyxy
        else:
            return self.func(frame, **kwds)
