from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, TypeVar

QueryType = TypeVar("QueryType", bound="Query")


class Node(metaclass=ABCMeta):
    """
    Nodes are the base of the Query API.

    They accept an iterable of frames and output another iterable
    after the operation has been performed.
    """

    @abstractmethod
    def __call__(self, target: Iterable) -> Iterable:
        pass


@dataclass(frozen=True)
class MapNode(Node):
    udf: Callable

    def __call__(self, target: Iterable) -> Iterable:
        for item in target:
            yield self.udf(item)


@dataclass(frozen=True)
class FilterNode(Node):
    condition: Callable

    def __call__(self, target: Iterable) -> Iterable:
        for item in target:
            if self.condition(item):
                yield item


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

    def __call__(self, target: Iterable) -> Iterable:
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
