from os import PathLike
from pathlib import Path
from typing import List, Optional, Type, TypeVar
from enum import Enum, auto

import cv2  # type: ignore

from tiny_eva.frame import Frame


VideoType = TypeVar("VideoType", bound="Video")


class VideoSource(Enum):
    MP4_FILE = auto()
    FRAME_LIST = auto()


class Video:
    def __init__(
        self,
        video_source: VideoSource,
        mp4_file: Optional[PathLike] = None,
        frame_list: Optional[List[Frame]] = None,
    ) -> None:
        """
        Represents a .mp4 video whose source is located at the path the user passed in.

        Arguments:
            video_source: enum storing whether video source is mp4 file
            on disk or frame list in memory
            mp4_file: Path to mp4 file. Only defined if video is stored as mp4 file.
            frame_list: List of Frame objects. Only defined if video is stored in memory.
        """
        self._source = video_source
        self._mp4_file: Optional[Path] = None
        self._frame_list: Optional[List[Frame]] = None
        self.frames_path: Optional[Path] = None
        self.is_decoded: bool = False
        self._num_frames = -1

        if self._source == VideoSource.MP4_FILE:
            self._mp4_file = Path(mp4_file) if mp4_file is not None else None
        elif self._source == VideoSource.FRAME_LIST:
            self._frame_list = frame_list
            self._num_frames = len(frame_list) if frame_list is not None else -1

    @classmethod
    def from_mp4_file(cls: Type[VideoType], source: PathLike) -> VideoType:
        """
        Creates Video from mp4 file stored on disk.
        The user must run decode() before accessing individual frames.

        Arguments:
            source: path to mp4 file
        """
        return cls(video_source=VideoSource.MP4_FILE, mp4_file=source)

    @classmethod
    def from_frames(cls: Type[VideoType], frames: List[Frame]) -> VideoType:
        """
        Creates Video from in-memory collection of frames.
        These frames can either be stored on disk as jpeg images or
        stored in-memory as numpy arrays. See Frame for more information.

        Arguments:
            frames: list of Frame
        """
        return cls(video_source=VideoSource.FRAME_LIST, frame_list=frames)

    def _frame_name(self, idx: int) -> str:
        return f"frame{idx}.jpg"

    def decode(self, destination_dir: PathLike) -> None:
        """
        Decodes the .mp4 video into jpeg frames and save to directory the user passes in.
        TODO: handle frame list case

        Arguments:
            destination_dir: path to directory to store decoded frames

        Raises:
            ValueError: if no mp4 file is found at source path
        """
        if self._mp4_file is None or not self._mp4_file.exists():
            raise ValueError("No mp4 source file found.")

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
        if self._source == VideoSource.FRAME_LIST:
            if self._frame_list is None:
                return -1
            return len(self._frame_list)

        if not self.is_decoded:
            raise ValueError("Must decode video with decode() before calculating len()")

        return self._num_frames

    def __getitem__(self, idx: int) -> Frame:
        """
        After video is decoded, individual frames can be accessed with idx.

        Arguments:
            idx: index of frame to access. Must be in bounds of video length

        Returns:
            np.array of shape (num_channels, height, width)

        Raises:
            IndexError: Index out of bounds or negative index
            ValueError: Video source is not found. This is either due to
            frame list being missing or Video not being decoded.
        """
        if idx < 0:
            raise IndexError("Negative indexing not suppoerted at this time")

        if self._source == VideoSource.FRAME_LIST:
            if self._frame_list is None:
                raise ValueError("Frame list not found")
            return self._frame_list[idx]

        if not self.is_decoded:
            raise ValueError("You must decode the video first before accessing frames.")

        if idx >= self._num_frames:
            raise IndexError(
                f"Cannot get frame {idx}.\nThere are {self._num_frames} frames in the video."
            )

        return Frame.from_source(
            (self.frames_path or Path.home()) / self._frame_name(idx)
        )
