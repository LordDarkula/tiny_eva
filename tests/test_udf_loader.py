import numpy as np  # type: ignore
import pytest

from tiny_eva.udf_loader import UDF
from tiny_eva.frame import Frame


@pytest.fixture(scope="session")
def yolo():
    return UDF.from_torch_hub("ultralytics/yolov5", "yolov5s", pretrained=True)


@pytest.fixture(scope="session")
def alexnet():
    return UDF.from_torch_hub("pytorch/vision", "alexnet", pretrained=True)


def test_all_zeros_true_on_zeros_frame(all_zeros_udf):
    sample_frame = Frame.from_numpy(np.zeros((3, 2, 3)))
    assert all_zeros_udf(sample_frame)


def test_all_zeros_false_in_on_ones_frame(all_zeros_udf):
    sample_frame = Frame.from_numpy(np.ones((3, 2, 3)))
    assert not all_zeros_udf(sample_frame)


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


@pytest.mark.skip
def test_alexnet(alexnet, traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    result = alexnet(frame)
    assert len(result) == 1
