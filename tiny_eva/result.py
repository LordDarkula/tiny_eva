from dataclasses import dataclass

from tiny_eva.bbox import Bbox


@dataclass(frozen=True)
class Result:
    pred_class: str
    bbox: Bbox
