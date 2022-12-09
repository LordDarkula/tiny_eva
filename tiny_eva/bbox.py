from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type, TypeVar

BboxType = TypeVar("BboxType", bound="Bbox")


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
    label: Optional[Any] = None

    @classmethod
    def from_coords(
        cls: Type[BboxType],
        c1: Tuple[int, int],
        c2: Tuple[int, int],
        label: Optional[str] = None,
    ) -> BboxType:
        """
        Construct Bbox from two tuples of two ints each representing two corners.

        Arguments:
            c1: bottom-left corner
            c2: top-right corner
        """
        return cls(x1=c1[0], y1=c1[1], x2=c2[0], y2=c2[1], label=label)

    @property
    def width(self) -> int:
        return abs(self.x1 - self.x2)

    @property
    def height(self) -> int:
        return abs(self.y1 - self.y2)
