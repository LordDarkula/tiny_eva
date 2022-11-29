from abc import abstractmethod, ABCMeta
from typing import Any

from tiny_eva.frame import AbstractFrame


class GenericLoader(metaclass=ABCMeta):
    """
    Defines interface for all UDF loaders.
    """

    @abstractmethod
    def __call__(self, frame: AbstractFrame, **kwds: Any) -> Any:
        """Run model on frame and return result."""
