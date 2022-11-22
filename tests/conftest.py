from pathlib import Path

import pytest
import numpy as np  # type: ignore

import tiny_eva
from tiny_eva.frame import Frame, AbstractFrame
from tiny_eva.video import Video
from tiny_eva.result import Result
from tiny_eva.bbox import Bbox
from tiny_eva.udf_loader import UDF


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
    def all_zeros(frame: AbstractFrame) -> Result:
        np_frame = frame.to_numpy()
        frame_shape = np_frame.shape  # type: ignore
        res = not np.any(np_frame)
        return Result(
            result=res,
            frame=Frame.from_numpy(np_frame),
            bboxes={Bbox(x1=0, y1=0, x2=frame_shape[2], y2=frame_shape[1], label=res)},
        )

    return UDF.from_callable(all_zeros)
