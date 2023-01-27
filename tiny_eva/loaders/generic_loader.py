from abc import ABCMeta, abstractmethod
from typing import Any

from tiny_eva.frame import GenericFrame


class GenericLoader(metaclass=ABCMeta):
    """
    Defines interface for all UDF loaders.
    """

    @abstractmethod
    def __call__(self, frame: GenericFrame, **kwds: Any) -> Any:
        """Run model on frame and return result."""
