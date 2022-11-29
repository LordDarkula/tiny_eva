from dataclasses import dataclass
from typing import List, Any
from abc import ABCMeta, abstractmethod

from tiny_eva.bbox import Bbox


@dataclass(frozen=True)
class Result(metaclass=ABCMeta):
    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@dataclass(frozen=True)
class MultiResult(Result):
    bboxes: List[Bbox]

    def __bool__(self) -> bool:
        return len(self.bboxes) != 0

    def __len__(self) -> int:
        return len(self.bboxes)


@dataclass(frozen=True)
class SingularResult(Result):
    result: Any

    def __bool__(self) -> bool:
        return bool(self.result)

    def __int__(self) -> int:
        return int(self.result)

    def __len__(self) -> int:
        return 1
