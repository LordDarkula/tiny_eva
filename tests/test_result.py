import pytest

from tiny_eva.result import Result
from tiny_eva.bbox import Bbox
from tiny_eva.frame import Frame


@pytest.fixture
def traffic_result(traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    bboxes = {
        Bbox.from_coords(c1=(0, 0), c2=(20, 30), label="car"),
        Bbox.from_coords(c1=(32, 41), c2=(58, 55), label="tree"),
        Bbox.from_coords(c1=(3, 41), c2=(44, 45), label="car"),
    }
    return Result(frame=Frame, bboxes=bboxes)


def test_two_cars_in_result(traffic_result):
    assert traffic_result.label_count("car") == 2
