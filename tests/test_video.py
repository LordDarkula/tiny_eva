from pathlib import Path
import os

import pytest

import tiny_eva
from tiny_eva.video import Video


@pytest.fixture(scope="session")
def decoded_video(tmp_path_factory):
    package_path = Path(tiny_eva.__file__).parent
    sample_video_location = package_path.parent / "data" / "sample_640x360.mp4"
    decode_location = tmp_path_factory.mktemp("out")

    video = Video(sample_video_location.resolve())
    video.decode(decode_location)
    return video


def test_decoded_num_frames(decoded_video):
    assert len(decoded_video) == 400


def test_decoded_all_frames_jpg(decoded_video):
    for fname in os.listdir(decoded_video.frames_path):
        frame_path = os.path.join(decoded_video.frames_path, fname)
        if os.path.isfile(frame_path):
            ext = os.path.splitext(frame_path)[1]
            assert ext == ".jpg" or ext == ".jpeg"


@pytest.mark.parametrize("test_input", [(0), (3), (399)])
def test_query_valid_frame_shape(decoded_video, test_input):
    frame = decoded_video[test_input]
    assert frame.to_numpy().shape == (3, 360, 640)


@pytest.mark.parametrize("test_input", [(-1), (400), (523)])
def test_query_out_of_bounds(decoded_video, test_input):
    with pytest.raises(IndexError):
        decoded_video[test_input]


def test_iterate_over_video(decoded_video):
    count = 0
    for frame in decoded_video:
        if count == 3:
            break
        assert frame.to_numpy().shape == (3, 360, 640)

        count += 1
