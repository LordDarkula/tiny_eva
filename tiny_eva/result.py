from dataclasses import dataclass


@dataclass(frozen=True)
class Bbox:
    """
    BBox stores the bounding box for an object detected by some object detection model.
    (x1, y1) is the coordinate of the bottom-left of the box
    (x2, y2) is the coordinate of the top-right of the box
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def length(self) -> int:
        return abs(self.x1 - self.x2)

    @property
    def width(self) -> int:
        return abs(self.y1 - self.y2)


@dataclass(frozen=True)
class Result:
    pred_class: str
    bbox: Bbox
