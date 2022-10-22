from os import PathLike
from pathlib import Path


class Video:
    def __init__(self, source: PathLike) -> None:
        self.source = Path(source)

    def decode(self, destination: PathLike) -> None:
        pass
