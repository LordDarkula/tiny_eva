from typing import Callable


from tiny_eva.loaders.torch_hub_loader import TorchHubLoader
from tiny_eva.loaders.callable_loader import CallableLoader


class UDF:
    @staticmethod
    def from_callable(func: Callable) -> CallableLoader:
        """
        Create UDF from Callable that returns subclass of Result.

        Arguments:
            func: callable that accepts Frame and returns Result
        """
        return CallableLoader(func)

    @staticmethod
    def from_torch_hub(
        model_uri: str, name: str, pretrained: bool = True
    ) -> TorchHubLoader:
        """
        Loads model from pytorch hub. Downloads model, so this will take time.

        Arguments:
            model_uri: hub path to model repo
            name: name of hub model
            pretrained: should fetched model be pretrained
        """
        return TorchHubLoader(model_uri, name, pretrained=pretrained)
