from typing import Any, Callable
from enum import Enum, auto
from abc import ABCMeta, abstractmethod

import torch  # type: ignore

from tiny_eva.frame import Frame


class AbstractLoader(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, frame: Frame, **kwds: Any) -> Any:
        """Run model on frame and return result."""


class CallableLoader(AbstractLoader):
    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(self, frame: Frame, **kwds: Any) -> Any:
        return self._func(frame, **kwds)


class TorchHubLoader(AbstractLoader):
    def __init__(self, model_uri: str, name: str, **kwds: Any) -> None:
        self._torch_model = torch.hub.load(model_uri, name, **kwds)

    def __call__(self, frame: Frame, **kwds: Any) -> Any:
        return self._torch_model(frame.source).pandas().xyxy


class UDF:
    @staticmethod
    def from_callable(func: Callable):
        return CallableLoader(func)

    @staticmethod
    def from_torch_hub(model_uri: str, name: str, pretrained: bool = True):
        """
        Loads model from pytorch hub. Downloads model, so this will take time.

        Arguments:
            model_uri: hub path to model repo
            name: name of hub model
            pretrained: should fetched model be pretrained
        """
        return TorchHubLoader(model_uri, name, pretrained=pretrained)
