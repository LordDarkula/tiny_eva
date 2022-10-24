from os import PathLike
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
from numpy.typing import ArrayLike  # type: ignore
from PIL import Image  # type: ignore


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

    @classmethod
    def from_source(cls, source: PathLike):
        """
        Accepts path to a valid jpeg image
        """
        return cls(source=source)

    @classmethod
    def from_numpy(cls, frame_array: ArrayLike):
        """
        Accepts np.array of shape (num_channels, height, width)
        """
        return cls(frame_array=frame_array)

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


class Video:
    def __init__(self, source: PathLike) -> None:
        """
        Represents a .mp4 video whose source is located at the path the user passed in.
        """
        self.source = Path(source)
        self.frames_path: Optional[Path] = None
        self.is_decoded: bool = False
        self._num_frames = -1

    def _frame_name(self, idx: int) -> str:
        return f"frame{idx}.jpg"

    def decode(self, destination_dir: PathLike) -> None:
        """
        Decodes the .mp4 video into jpeg frames and save to directory the user passes in.
        """
        self.frames_path = Path(destination_dir)
        self.frames_path.mkdir(exist_ok=True)

        vidcap = cv2.VideoCapture(str(self.source.resolve()))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                f"{self.frames_path.resolve()}/{self._frame_name(count)}", image
            )
            success, image = vidcap.read()
            print(f"Read a new frame: {success}")
            count += 1

        self.is_decoded = True
        self._num_frames = count

    def __len__(self) -> int:
        if not self.is_decoded:
            return -1
        return self._num_frames

    def __getitem__(self, idx: int) -> Frame:
        """
        After video is decoded, individual frames can be accessed with idx.

        Arguments:
            idx: index of frame to access. Must be in bounds of video length

        Returns:
            np.array of shape (num_channels, height, width)
        """
        if not self.is_decoded:
            raise ValueError("You must decode the video first before accessing frames.")

        if idx < 0 or idx >= self._num_frames:
            raise IndexError(
                f"Cannot get frame {idx}.\nThere are {self._num_frames} frames in the video."
            )

        return Frame(source=(self.frames_path or Path.home()) / self._frame_name(idx))
