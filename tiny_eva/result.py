from dataclasses import dataclass
from typing import Iterable, Any, Optional

from tiny_eva.bbox import Bbox
from tiny_eva.frame import AbstractFrame


@dataclass(frozen=True)
class Result:
    frame: AbstractFrame
    bboxes: Iterable[Bbox]
    result: Optional[Any] = None

    def label_count(self, label: str):
        return len([None for box in self.bboxes if box.label == label])

    def __bool__(self) -> bool:
        return bool(self.result)

    def __len__(self) -> int:
        return len(self.bboxes)
