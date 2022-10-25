from dataclasses import dataclass


@dataclass(frozen=True)
class Bbox:
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
