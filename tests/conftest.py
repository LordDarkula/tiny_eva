from pathlib import Path

import pytest

import tiny_eva


@pytest.fixture
def traffic_frame_path():
    relative = Path(tiny_eva.__file__).parent.parent / "data" / "traffic.jpeg"
    yield relative.resolve()
