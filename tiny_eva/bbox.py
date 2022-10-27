from dataclasses import dataclass
from typing import Tuple


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

    @classmethod
    def from_coords(cls, c1: Tuple[int, int], c2: Tuple[int, int]):
        return cls(x1=c1[0], y1=c1[1], x2=c2[0], y2=c2[1])

    @property
    def length(self) -> int:
        return abs(self.x1 - self.x2)

    @property
    def width(self) -> int:
        return abs(self.y1 - self.y2)
