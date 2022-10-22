from os import PathLike
from pathlib import Path
from typing import Optional

import cv2  # type: ignore


class Video:
    def __init__(self, source: PathLike) -> None:
        self.source = Path(source)
        self.frames_path: Optional[Path] = None
        self._num_frames = -1

    def decode(self, destination_dir: PathLike) -> None:
        self.frames_path = Path(destination_dir)
        self.frames_path.mkdir(exist_ok=True)

        vidcap = cv2.VideoCapture(str(self.source.resolve()))
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{self.frames_path.resolve()}/frame{count}.jpg", image)
            success, image = vidcap.read()
            print("Read a new frame: ", success)
            count += 1
        self._num_frames = count

    def __len__(self) -> int:
        return self._num_frames
