from os import PathLike
from pathlib import Path
from typing import List, Optional
from enum import Enum, auto

import cv2  # type: ignore

from tiny_eva.frame import Frame


class VideoSource(Enum):
    MP4_FILE = auto()
    FRAME_LIST = auto()


class Video:
    def __init__(self, mp4_file: Optional[PathLike] = None) -> None:
        """
        Represents a .mp4 video whose source is located at the path the user passed in.
        """

        self._mp4_file: Optional[PathLike] = (
            Path(mp4_file) if mp4_file is not None else None
        )
        self.frames_path: Optional[Path] = None
        self.is_decoded: bool = False
        self._num_frames = -1

    def from_frames(cls, frames: List[Frame]):
        pass

    def _frame_name(self, idx: int) -> str:
        return f"frame{idx}.jpg"

    def decode(self, destination_dir: PathLike) -> None:
        """
        Decodes the .mp4 video into jpeg frames and save to directory the user passes in.
        """
        self.frames_path = Path(destination_dir)
        self.frames_path.mkdir(exist_ok=True)

        vidcap = cv2.VideoCapture(str(self._mp4_file.resolve()))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                f"{self.frames_path.resolve()}/{self._frame_name(count)}", image
            )
            success, image = vidcap.read()
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
