from typing import Any

import torch  # type: ignore

from tiny_eva.loaders.generic_loader import GenericLoader
from tiny_eva.frame import AbstractFrame


class TorchHubLoader(GenericLoader):
    """
    Create UDF from model loaded from pytorch hub.

    Downloads model from torch hub at object initialization time.
    """

    def __init__(self, model_uri: str, name: str, **kwds: Any) -> None:
        """
        Download model from torch hub.

        Arguments:
            model_uri: hub path to model repo
            name: name of hub model
            pretrained: should fetched model be pretrained
        """
        self._torch_model = torch.hub.load(model_uri, name, **kwds)

    def __call__(self, frame: AbstractFrame, **kwds: Any) -> Any:
        """
        TODO make function return Result instance
        """
        return self._torch_model(frame.to_numpy()).pandas().xyxy
