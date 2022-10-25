from tiny_eva.result import Bbox


def test_bbox_length():
    bbox = Bbox(x1=0, y1=0, x2=50, y2=50)
    assert bbox.length == 50


def test_bbox_width():
    bbox = Bbox(x1=0, y1=0, x2=50, y2=50)
    assert bbox.width == 50
