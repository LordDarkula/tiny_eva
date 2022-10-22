from pathlib import Path
import os

import tiny_eva
from tiny_eva.decoder import Video


def test_decoded_num_frames(tmp_path):
    package_path = Path(tiny_eva.__file__).parent
    sample_video_location = package_path.parent / "data" / "sample_640x360.mp4"
    print(f"{package_path=}")

    video = Video(sample_video_location)
    video.decode(tmp_path)

    assert len(video) == 400


def test_decoded_all_frames_jpg(tmp_path):
    package_path = Path(tiny_eva.__file__).parent
    sample_video_location = package_path / "data" / "sample_640x360.mp4"
    print(f"{package_path=}")

    video = Video(sample_video_location)
    video.decode(tmp_path)

    for fname in os.listdir(tmp_path):
        frame_path = os.path.join(tmp_path, fname)
        if os.path.isfile(frame_path):
            ext = os.path.splitext(frame_path)[1]
            assert ext == ".jpg" or ext == ".jpeg"
