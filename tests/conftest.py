from pathlib import Path

import pytest
import numpy as np  # type: ignore

import tiny_eva
from tiny_eva.frame import Frame
from tiny_eva.video import Video


@pytest.fixture
def traffic_frame_path():
    relative = Path(tiny_eva.__file__).parent.parent / "data" / "traffic.jpeg"
    yield relative.resolve()


@pytest.fixture(scope="session")
def blank_frame_video():
    frame_arr = np.zeros((3, 20, 20))
    frames = [Frame.from_numpy(frame_arr) for _ in range(15)]
    video = Video.from_frames(frames)
    return video
