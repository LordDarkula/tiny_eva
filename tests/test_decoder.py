from pathlib import Path
import os

import tiny_eva
from tiny_eva.decoder import Video


def test_recorder():
    package_path = Path(tiny_eva.__file__).parent
    sample_video_location = package_path / "data" / "sample_640x360.mp4"
    decoded_frames_location = package_path / "out"

    video = Video(sample_video_location)
    video.decode(decoded_frames_location)

    for fname in os.listdir(decoded_frames_location):
        if os.path.isfile(os.path.join(decoded_frames_location, fname)):
            ext = os.path.splitext[1]
            assert ext == ".jpg" or ext == ".jpeg"
