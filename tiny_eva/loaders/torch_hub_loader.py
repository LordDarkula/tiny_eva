from typing import Any

import torch  # type: ignore

from tiny_eva.frame import GenericFrame
from tiny_eva.loaders.generic_loader import GenericLoader
from tiny_eva.result import PandasResult


class TorchHubLoader(GenericLoader):
    """
    Create UDF from model loaded from pytorch hub.

    Model must be downloaded with download() method.
    """

    def __init__(self, model_uri: str, name: str, **kwds: Any) -> None:
        """
        Download model from torch hub.

        Arguments:
            model_uri: hub path to model repo
            name: name of hub model
            pretrained: should fetched model be pretrained
        """
        self._model_uri = model_uri
        self._model_name = name
        self._model_keyword_arguments = kwds
        self._torch_model = None

    def download(self) -> None:
        """
        Download model from torch hub
        """
        self._torch_model = torch.hub.load(
            self._model_uri, self._model_name, **self._model_keyword_arguments
        )

    def __call__(self, frame: GenericFrame, **kwds: Any) -> Any:
        """
        Evaluate downloaded model on a single frame.
        """
        if self._torch_model is None:
            raise ValueError("Must download model from Torch Hub before calling")

        bboxes = self._torch_model(frame.to_numpy()).pandas().xyxy[0]
        return PandasResult(bboxes)
