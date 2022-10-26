import numpy as np  # type: ignore
import pytest

from tiny_eva.loader import UDF
from tiny_eva.result import Result, Bbox
from tiny_eva.frame import Frame


@pytest.fixture
def all_zeros():
    def all_zeros(frame: Frame) -> Result:
        np_frame = frame.to_numpy()
        frame_shape = np_frame.shape  # type: ignore
        res = str(not np.any(np_frame))
        return Result(res, Bbox(x1=0, y1=0, x2=frame_shape[2], y2=frame_shape[1]))

    yield UDF.from_callable(all_zeros)


@pytest.fixture(scope="session")
def yolo():
    return UDF.from_torch_hub("ultralytics/yolov5", "yolov5s", pretrained=True)


@pytest.fixture(scope="session")
def alexnet():
    return UDF.from_torch_hub("pytorch/vision", "alexnet", pretrained=True)


def test_all_zeros_true(all_zeros):
    sample_frame = Frame.from_numpy(np.zeros((3, 2, 3)))
    assert all_zeros(sample_frame).pred_class == "True"


def test_all_zeros_false(all_zeros):
    sample_frame = Frame.from_numpy(np.ones((3, 2, 3)))
    assert not all_zeros(sample_frame) == "False"


def test_yolo_result_len(yolo, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = yolo(frame)
    assert len(result) == 1


@pytest.mark.skip
def test_yolo_first_result(yolo, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = yolo(frame)

    result_classes = {"car", "person", "tree", "truck"}

    assert result[0].pred_class in result_classes


# def test_alexnet(alexnet, traffic_frame_path):
#     frame = Frame.from_source(traffic_frame_path)
#     result = alexnet(frame)
#     assert len(result) == 1
