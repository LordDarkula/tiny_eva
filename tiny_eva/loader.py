from keyword import kwlist
from typing import Callable, Any


class UDF:
    def __init__(self, func: Callable) -> None:
        self.func: Callable = func

    @classmethod
    def from_callable(cls, func: Callable):
        return cls(func)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
