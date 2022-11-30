import numpy as np  # type: ignore
import pytest

from tiny_eva.frame import Frame, GenericFrame, JPEGFrame, NumpyFrame


@pytest.fixture
def traffic_jpeg_frame(traffic_frame_path):
    return JPEGFrame(traffic_frame_path)


@pytest.fixture
def traffic_np_frame(traffic_jpeg_frame):
    np_frame = traffic_jpeg_frame.to_numpy()
    return NumpyFrame(np_frame)


@pytest.fixture
def traffic2_jpeg_frame(traffic2_frame_path):
    return JPEGFrame(traffic2_frame_path)


def test_initialize_abstract_frame():
    with pytest.raises(TypeError):
        GenericFrame()


def test_initialize_jpeg_frame(traffic_jpeg_frame):
    assert traffic_jpeg_frame.to_numpy().shape == (3, 853, 1280)


def test_jpeg_frame_equality_same_frame(traffic_jpeg_frame):
    assert traffic_jpeg_frame == traffic_jpeg_frame


def test_jpeg_frame_equality_two_different_frames(
    traffic_jpeg_frame, traffic2_jpeg_frame
):
    assert traffic_jpeg_frame != traffic2_jpeg_frame


def test_jpeg_frame_equality_invalid_object(traffic_jpeg_frame):
    assert traffic_jpeg_frame != {"randomset"}


def test_frame_equality_jpeg_and_numpy_true(traffic_jpeg_frame, traffic_np_frame):
    assert traffic_jpeg_frame == traffic_np_frame


def test_frame_equality_jpeg_and_numpy_true_reversed(
    traffic_jpeg_frame, traffic_np_frame
):
    assert traffic_np_frame == traffic_jpeg_frame


def test_frame_equality_jpeg_and_numpy_false(traffic2_jpeg_frame, traffic_np_frame):
    assert traffic2_jpeg_frame != traffic_np_frame


def test_frame_equality_jpeg_and_numpy_false_reversed(
    traffic2_jpeg_frame, traffic_np_frame
):
    assert traffic_np_frame != traffic2_jpeg_frame


def test_initialize_numpy_frame():
    np_frame = np.ones((3, 200, 300))
    frame = NumpyFrame(np_frame)
    assert np.array_equal(frame.to_numpy(), np_frame)


def test_initialize_from_array():
    np_frame = np.ones((3, 200, 300))
    frame = Frame.from_numpy(np_frame)
    assert np.array_equal(frame.to_numpy(), np_frame)


def test_intialize_from_source(traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    assert frame.to_numpy().shape == (3, 853, 1280)
