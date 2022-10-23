from typing import Callable, Any


class UDF:
    @classmethod
    def from_callable(cls, func: Callable):
        return cls()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return True
