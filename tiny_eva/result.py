from dataclasses import dataclass
from typing import Iterable

from tiny_eva.bbox import Bbox
from tiny_eva.frame import AbstractFrame


@dataclass(frozen=True)
class Result:
    frame: AbstractFrame
    bboxes: Iterable[Bbox]

    def label_count(self, label: str):
        return len([None for box in self.bboxes if box.label == label])
