import pytest

from tiny_eva.loaders.udf_loader import UDF
from tiny_eva.frame import Frame


@pytest.fixture(scope="session")
def yolo():
    return UDF.from_torch_hub("ultralytics/yolov5", "yolov5s", pretrained=True)


@pytest.fixture(scope="session")
def alexnet():
    return UDF.from_torch_hub("pytorch/vision", "alexnet", pretrained=True)


@pytest.fixture
def test_yolo_result_len(yolo, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = yolo(frame)
    assert len(result) == 1


def test_yolo_first_result(yolo, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = yolo(frame)
    assert result[0].label == "car"


@pytest.mark.skip
def test_alexnet(alexnet, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = alexnet(frame)
    assert len(result) == 1
