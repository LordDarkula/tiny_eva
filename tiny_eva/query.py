from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, TypeVar, Union

from tiny_eva.result import Result
from tiny_eva.video import Video

QueryType = TypeVar("QueryType", bound="Query")


class Node(metaclass=ABCMeta):
    """
    Nodes are the base of the Query API.

    They accept an iterable of frames and output another iterable
    after the operation has been performed.
    """

    @abstractmethod
    def __call__(
        self, target: Union[Video, List[Result]]
    ) -> Union[Video, List[Result]]:
        pass


@dataclass(frozen=True)
class MapNode(Node):
    udf: Callable

    def __call__(self, target: Union[Video, List[Result]]) -> List[Result]:
        if isinstance(target, Video):
            results = [self.udf(frame) for frame in target]
        else:
            results = [self.udf(result.frame) for result in target]

        return results


@dataclass(frozen=True)
class FilterNode(Node):
    condition: Callable

    def __call__(self, target: Union[Video, List[Result]]) -> Video:
        if isinstance(target, Video):
            frames = [frame for frame in target if self.condition(frame)]
        else:
            frames = [result.frame for result in target if self.condition(result.frame)]

        return Video.from_frames(frames)


@dataclass(frozen=True)
class ReduceNode(Node):
    condition: Callable

    def _recursive_call(self, target: List) -> Iterable:
        if len(target) == 1:
            return target[0]

        return self.condition(target[0], self._recursive_call(target[1:]))

    def __call__(self, target: Iterable) -> Iterable:
        if len(list(target)) == 1:
            return target

        return [self._recursive_call(list(target))]


@dataclass(frozen=True)
class Condition:
    """
    Condition allows the user to compare the Result of a UDF to
    some expected bound.

    Arguments:
        udf: Callable that returns an instance of Result
        result: Callable that takes Result and returns an instance of bool
    """

    udf: Callable
    result: Callable

    def __call__(self, item: Any) -> bool:
        return self.result(self.udf(item))


class Query:
    """
    Query is a functional-like API for applying UDFs to videos.

    Queries can be constructed with method chaining.

    >>> query = Query().map(lambda x: 2*x)
    >>> len(query)
    1
    """

    def __init__(self: QueryType) -> None:
        self._node_list: List[Node] = []

    def __len__(self) -> int:
        return len(self._node_list)

    def __call__(self, target: Video) -> Result:
        current = target
        for node in self._node_list:
            current = node(current)

        return current

    def map(self: QueryType, udf: Any) -> QueryType:
        self._node_list.append(MapNode(udf))
        return self

    def filter(self: QueryType, condition: Any) -> QueryType:
        self._node_list.append(FilterNode(condition))
        return self

    def reduce(self: QueryType, condition: Any) -> QueryType:
        self._node_list.append(ReduceNode(condition))
        return self
