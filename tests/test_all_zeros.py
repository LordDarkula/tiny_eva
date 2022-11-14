import pytest
import numpy as np

from tiny_eva.frame import Frame
from tiny_eva.video import Video


@pytest.fixture
def dummy_video():
    frames = [Frame.from_numpy(np.zeros((3, 20, 20)))]
    return Video.from_frames(frames)


@pytest.mark.skip
def test_simple_query():
    all_zeros = all_zeros_udf()
    query = Query().filter(all_zeros).all()
    executor = NaiveExecutor()
    executor.execute(frames, query)
