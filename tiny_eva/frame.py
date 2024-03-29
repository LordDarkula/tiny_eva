from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import Optional

import numpy as np  # type: ignore
from numpy.typing import ArrayLike  # type: ignore
from PIL import Image  # type: ignore


class GenericFrame(metaclass=ABCMeta):
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, GenericFrame):
            return False
        return np.array_equal(self.to_numpy(), __o.to_numpy())

    @abstractmethod
    def to_numpy(self) -> ArrayLike:
        """Convert Frame into numpy array.

        Returns:
            np.array of shape (num_channels, height, width)
        """


class JPEGFrame(GenericFrame):
    def __init__(self, frame_path: PathLike) -> None:
        self._frame_path = frame_path
        self._np_frame: Optional[ArrayLike] = None

    def to_numpy(self) -> ArrayLike:
        """Convert Frame into numpy array.

         Opens JPEG in file with PIL and converts into numpy array.

        Returns:
            np.array of shape (num_channels, height, width)
        """
        if self._np_frame is not None:
            return self._np_frame

        image = Image.open(self._frame_path)
        frame = np.asarray(image)
        channel_order = (2, 0, 1)
        self._np_frame = np.transpose(frame, channel_order)
        return self._np_frame


class NumpyFrame(GenericFrame):
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

    @staticmethod
    def from_source(source: PathLike) -> JPEGFrame:
        """
        Accepts path to a valid jpeg image
        """
        return JPEGFrame(source)

    @staticmethod
    def from_numpy(frame_array: ArrayLike) -> NumpyFrame:
        """
        Create Frame object from a numpy array.

        Arguments:
            frame_array: np.array of shape (num_channels, height, width)
        """
        return NumpyFrame(frame_array)
