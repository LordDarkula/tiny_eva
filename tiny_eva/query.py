from typing import Any, TypeVar, List, Callable, Iterable

from abc import abstractmethod, ABCMeta

QueryType = TypeVar("QueryType", bound="Query")


class AbstractNode(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, target: Any) -> Any:
        pass


class MapNode(AbstractNode):
    def __init__(self, udf: Any) -> None:
        self.udf = udf

    def __call__(self, target: Any) -> Any:
        return self.udf(target)


class FilterNode(AbstractNode):
    def __init__(self, condition: Callable) -> None:
        self.condition = condition

    def __call__(self, target: Any) -> bool:
        return self.condition(target)


class Query:
    """
    Query is a functional-like API for applying UDFs to videos.

    Queries can be constructed with method chaining.

    >>> query = Query().map(lambda x: 2*x)
    >>> len(query)
    1
    """

    def __init__(self: QueryType) -> None:

        self._node_list: List[AbstractNode] = []

    def __len__(self):
        return len(self._node_list)

    def __call__(self, target: Iterable) -> Iterable:
        for item in target:
            result = item
            for node in self._node_list:
                result = node(item)

            yield result

    def map(self: QueryType, udf: Any) -> QueryType:
        self._node_list.append(MapNode(udf))
        return self

    def filter(self: QueryType, condition: Any) -> QueryType:
        self._node_list.append(FilterNode(condition))
        return self
