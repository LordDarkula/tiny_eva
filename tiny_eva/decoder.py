from os import PathLike
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
from numpy.typing import ArrayLike  # type: ignore
from PIL import Image  # type: ignore


class Video:
    def __init__(self, source: PathLike) -> None:
        """
        Represents a .mp4 video whose source is located at the path the user passed in.
        """
        self.source = Path(source)
        self.frames_path: Optional[Path] = None
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
        self._num_frames = count

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx: int) -> ArrayLike:
        """
        After video is decoded, individual frames can be accessed with idx.

        Arguments:
            idx: index of frame to access. Must be in bounds of video length

        Returns:
            np.array of shape (num_channels, height, width)
        """
        if self.frames_path is None:
            raise ValueError("You must decode the video first before accessing frames.")

        if idx < 0 or idx >= self._num_frames:
            raise IndexError(
                f"Cannot get frame {idx}.\nThere are {self._num_frames} frames in the video."
            )

        image = Image.open(self.frames_path / self._frame_name(idx))
        frame = np.asarray(image)
        return np.transpose(frame, (2, 0, 1))
