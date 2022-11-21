from os import PathLike
from pathlib import Path
from typing import Any, Optional, Type, TypeVar
from abc import ABCMeta, abstractmethod

import numpy as np  # type: ignore
from numpy.typing import ArrayLike  # type: ignore
from PIL import Image  # type: ignore

FrameType = TypeVar("FrameType", bound="Frame")


class AbstractFrame(metaclass=ABCMeta):
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, AbstractFrame):
            return False
        return np.array_equal(self.to_numpy(), __o.to_numpy())

    @abstractmethod
    def to_numpy(self) -> ArrayLike:
        """Convert Frame into numpy array.

        Returns:
            np.array of shape (num_channels, height, width)
        """


class JPEGFrame(AbstractFrame):
    def __init__(self, frame_path: PathLike) -> None:
        self._frame_path = frame_path

    def to_numpy(self) -> ArrayLike:
        """Convert Frame into numpy array.

         Opens JPEG in file with PIL and converts into numpy array.

        Returns:
            np.array of shape (num_channels, height, width)
        """
        image = Image.open(self._frame_path)
        frame = np.asarray(image)
        return np.transpose(frame, (2, 0, 1))


class NumpyFrame(AbstractFrame):
    def __init__(self, np_frame: ArrayLike) -> None:
        self._np_frame = np_frame

    def to_numpy(self) -> ArrayLike:
        """Convert Frame into numpy array.

         Returns in-memory numpy array.

        Returns:
            np.array of shape (num_channels, height, width)
        """
        return self._np_frame


class Frame:
    """
    Frame represents a single frame from a video.
    A Frame can be a jpeg image stored on disk or a numpy array in memory.
    A Frame must be either on-disk or in-memory but not both or neither.
    """

    def __init__(
        self, source: Optional[PathLike] = None, frame_array: Optional[ArrayLike] = None
    ):
        if source is None and frame_array is None:
            raise ValueError(
                "A Frame must be initialized with either a source path or a numpy array."
            )

        if source is not None and frame_array is not None:
            raise ValueError(
                "A Frame cannot be initializaed with both a source path and a numpy array."
            )

        self._source = Path(source) if source is not None else None
        self._frame_array = frame_array

    @staticmethod
    def from_source(source: PathLike) -> JPEGFrame:
        """
        Accepts path to a valid jpeg image
        """
        return JPEGFrame(source)

    @classmethod
    def from_numpy(cls: Type[FrameType], frame_array: ArrayLike) -> FrameType:
        """
        Create Frame object from a numpy array.

        Arguments:
            frame_array: np.array of shape (num_channels, height, width)
        """
        return cls(frame_array=frame_array)

    @property
    def source(self) -> Optional[PathLike]:
        return self._source

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Frame):
            return False

        return np.array_equal(self.to_numpy(), other.to_numpy())

    def to_numpy(self) -> ArrayLike:
        """
        Convert Frame into numpy array.

        Returns:
            np.array of shape (num_channels, height, width)
        """
        if self._frame_array is None:
            image = Image.open(self._source)
            frame = np.asarray(image)
            self._frame_array = np.transpose(frame, (2, 0, 1))

        return self._frame_array
