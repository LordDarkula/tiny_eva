import numpy as np  # type: ignore
import pytest

from tiny_eva.loader import UDF
from tiny_eva.decoder import Frame


@pytest.fixture
def all_zeros():
    def all_zeros(frame: Frame) -> bool:
        return not np.any(frame.to_numpy())

    yield UDF.from_callable(all_zeros)


@pytest.fixture
def yolo():
    yield UDF.from_torch_hub("ultralytics/yolov5", "yolov5s", pretrained=True)


def test_all_zeros_true(all_zeros):
    sample_frame = Frame.from_numpy(np.zeros((3, 2, 3)))
    assert all_zeros(sample_frame)


def test_all_zeros_false(all_zeros):
    sample_frame = Frame.from_numpy(np.ones((3, 2, 3)))
    assert not all_zeros(sample_frame)


def test_yolo(yolo, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = yolo(frame)
    assert len(result) > 0
