from typing import Any, TypeVar, List

QueryType = TypeVar("QueryType", bound="Query")
FinalizedQueryType = TypeVar("FinalizedQueryType", bound="FinalizedQuery")


class AbstractNode:
    child: Any


class MapNode(AbstractNode):
    def __init__(self, udf: Any) -> None:
        self.udf = udf


class FilterNode(AbstractNode):
    condition: Any


class Query:
    def __init__(self) -> None:

        self._node_list: List[AbstractNode] = []

    def __len__(self):
        return len(self._node_list)

    def map(self: QueryType, udf: Any) -> QueryType:
        self._node_list.append(MapNode(udf))
        return self

    def filter(self: QueryType, condition: Any) -> QueryType:
        return self

    def all(self: QueryType) -> FinalizedQueryType:
        return FinalizedQuery(self, "all")


class FinalizedQuery:
    query: QueryType
    finalizer: Any
