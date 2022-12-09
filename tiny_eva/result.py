from dataclasses import dataclass
from typing import List, Any
from abc import ABCMeta, abstractmethod

import pandas as pd  # type: ignore

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
class PandasResult(Result):
    df: pd.DataFrame

    def __bool__(self) -> bool:
        return self.df.empty

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Bbox:
        """
        Get the BBox associated with a specific result.

        Models can return multiple results each with it's own Bbox and class.

        Arguments:
            idx: index of result

        Returns:
            Bbox containing result
        """
        coord1x = int(self.df.loc[idx, "xmin"])
        coord1y = int(self.df.loc[idx, "ymin"])
        coord2x = int(self.df.loc[idx, "xmax"])
        coord2y = int(self.df.loc[idx, "ymax"])
        label = str(self.df.loc[idx, "name"])
        return Bbox.from_coords(
            c1=(coord1x, coord1y), c2=(coord2x, coord2y), label=label
        )


@dataclass(frozen=True)
class SingularResult(Result):
    result: Any

    def __bool__(self) -> bool:
        return bool(self.result)

    def __int__(self) -> int:
        return int(self.result)

    def __len__(self) -> int:
        return 1
