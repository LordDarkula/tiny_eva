import pytest

from tiny_eva.frame import Frame, GenericFrame
from tiny_eva.loaders.udf_loader import UDF


@pytest.fixture
def traffic_frame(traffic_frame_path) -> GenericFrame:
    return Frame.from_source(traffic_frame_path)


@pytest.fixture(scope="session")
def yolo():
    return UDF.from_torch_hub("ultralytics/yolov5", "yolov5s", pretrained=True)


@pytest.fixture(scope="session")
def undownloaded_alexnet():
    torch_hub_loader = UDF.from_torch_hub("pytorch/vision", "alexnet", pretrained=True)
    return torch_hub_loader


@pytest.fixture(scope="session")
def downloaded_alexnet(undownloaded_alexnet):
    undownloaded_alexnet.download()
    return undownloaded_alexnet


@pytest.fixture
def test_yolo_result_len(yolo, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = yolo(frame)
    assert len(result) == 1


@pytest.mark.skip
def test_yolo_first_result(yolo, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = yolo(frame)
    assert result[0].label == "car"


def test_alexnet_undownloaded(undownloaded_alexnet, traffic_frame):
    with pytest.raises(ValueError):
        undownloaded_alexnet(traffic_frame)


def test_alexnet(downloaded_alexnet, traffic_frame):
    result = downloaded_alexnet(traffic_frame)
    assert len(result) == 1
