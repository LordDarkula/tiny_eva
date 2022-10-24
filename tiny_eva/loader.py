from typing import Callable, Any
from os import PathLike


class UDF:
    def __init__(self, func: Callable) -> None:
        self.func: Callable = func

    @classmethod
    def from_callable(cls, func: Callable):
        return cls(func)

    @classmethod
    def from_torch_hub(cls, path: PathLike, name: str, pretrained: bool = True):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
