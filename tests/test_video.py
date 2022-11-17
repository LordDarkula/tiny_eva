from pathlib import Path
import os
import copy

import pytest

import tiny_eva
from tiny_eva.video import Video


@pytest.fixture(scope="session")
def beach_mp4():
    package_path = Path(tiny_eva.__file__).parent
    sample_video_location = package_path.parent / "data" / "sample_640x360.mp4"
    return Video.from_mp4_file(sample_video_location.resolve())


@pytest.fixture(scope="session")
def decoded_mp4_video(tmp_path_factory, beach_mp4):
    decode_location = tmp_path_factory.mktemp("out")
    beach_copy = copy.deepcopy(beach_mp4)
    beach_copy.decode(decode_location)
    return beach_copy


def test_raw_video_has_invalid_len(beach_mp4):
    with pytest.raises(ValueError):
        assert len(beach_mp4) == -1


def test_raw_video_invalid_index_access(beach_mp4):
    with pytest.raises(ValueError):
        beach_mp4[0]


def test_decoded_len(decoded_mp4_video):
    assert len(decoded_mp4_video) == 400


def test_decoded_all_frames_jpg(decoded_mp4_video):
    for fname in os.listdir(decoded_mp4_video.frames_path):
        frame_path = os.path.join(decoded_mp4_video.frames_path, fname)
        if os.path.isfile(frame_path):
            ext = os.path.splitext(frame_path)[1]
            assert ext == ".jpg" or ext == ".jpeg"


@pytest.mark.parametrize("test_input", [(0), (3), (399)])
def test_mp4_index_access_valid_frame_shape(decoded_mp4_video, test_input):
    frame = decoded_mp4_video[test_input]
    assert frame.to_numpy().shape == (3, 360, 640)


@pytest.mark.parametrize("test_input", [(-1), (400), (523)])
def test_mp4_index_access_out_of_bounds(decoded_mp4_video, test_input):
    with pytest.raises(IndexError):
        decoded_mp4_video[test_input]


def test_frame_list_len(blank_frame_video):
    assert len(blank_frame_video) == 15


@pytest.mark.parametrize("test_input", [(0), (3), (14)])
def test_frame_list_index_access_valid_frame_shape(blank_frame_video, test_input):
    frame = blank_frame_video[test_input]
    assert frame.to_numpy().shape == (3, 20, 20)


@pytest.mark.parametrize("test_input", [(-1), (15), (42)])
def test_frame_list_index_access_out_of_bounds(blank_frame_video, test_input):
    with pytest.raises(IndexError):
        blank_frame_video[test_input]


def test_iterate_over_video(decoded_mp4_video):
    count = 0
    for frame in decoded_mp4_video:
        if count == 3:
            break
        assert frame.to_numpy().shape == (3, 360, 640)

        count += 1
