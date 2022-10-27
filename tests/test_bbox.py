import pytest

from tiny_eva.result import Bbox


@pytest.fixture
def example_bbox():
    return Bbox(x1=0, y1=0, x2=50, y2=50)


def test_bbox_from_coords(example_bbox):
    assert Bbox.from_coords(c1=(0, 0), c2=(50, 50)) == example_bbox


def test_bbox_length(example_bbox):
    assert example_bbox.length == 50


def test_bbox_width(example_bbox):
    assert example_bbox.width == 50
