from pathlib import Path

import pytest
import numpy as np  # type: ignore

import tiny_eva
from tiny_eva.frame import Frame, GenericFrame
from tiny_eva.video import Video
from tiny_eva.result import SingularResult
from tiny_eva.loaders.udf_loader import UDF


@pytest.fixture
def traffic_frame_path():
    relative = Path(tiny_eva.__file__).parent.parent / "data" / "traffic.jpeg"
    yield relative.resolve()


@pytest.fixture
def traffic2_frame_path():
    relative = Path(tiny_eva.__file__).parent.parent / "data" / "traffic2.jpg"
    yield relative.resolve()


@pytest.fixture(scope="session")
def blank_frame_video():
    frame_arr = np.zeros((3, 20, 20))
    frames = [Frame.from_numpy(frame_arr) for _ in range(15)]
    video = Video.from_frames(frames)
    return video


@pytest.fixture
def all_zeros_udf():
    def all_zeros(frame: GenericFrame) -> SingularResult:
        np_frame = frame.to_numpy()
        res = not np.any(np_frame)
        return SingularResult(res)

    return UDF.from_callable(all_zeros)
