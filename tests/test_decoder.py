from pathlib import Path
import os

import pytest

import tiny_eva
from tiny_eva.decoder import Video


@pytest.fixture
def decoded_video(tmp_path):
    package_path = Path(tiny_eva.__file__).parent
    sample_video_location = package_path.parent / "data" / "sample_640x360.mp4"

    video = Video(sample_video_location)
    video.decode(tmp_path)
    yield video


def test_decoded_num_frames(decoded_video):
    assert len(decoded_video) == 400


def test_decoded_all_frames_jpg(decoded_video):
    for fname in os.listdir(decoded_video.frames_path):
        frame_path = os.path.join(decoded_video.frames_path, fname)
        if os.path.isfile(frame_path):
            ext = os.path.splitext(frame_path)[1]
            assert ext == ".jpg" or ext == ".jpeg"
