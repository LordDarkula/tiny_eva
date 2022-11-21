import numpy as np  # type: ignore
import pytest

from tiny_eva.frame import Frame, AbstractFrame, JPEGFrame, NumpyFrame


def test_initialize_abstract_frame():
    with pytest.raises(TypeError):
        AbstractFrame()


def test_initialize_jpeg_frame(traffic_frame_path):
    frame = JPEGFrame(traffic_frame_path)
    assert frame.to_numpy().shape == (3, 853, 1280)


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
