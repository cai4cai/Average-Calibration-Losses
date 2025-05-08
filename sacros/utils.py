from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import nn, optim
from torch.nn import functional as F

from monai.handlers import from_engine
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.config import KeysCollection
from monai.utils import ImageMetaKey as Key

from monai.transforms import AsDiscrete

__all__ = [
    "discrete_from_engine",
    "meta_data_batch_transform",
    "meta_data_batch_transform_dir",
    "meta_data_image_transform",
    "meta_data_image_transform_dir",
    "TemperatureScaling",
]


def discrete_from_engine(
    keys: Union[str, List[str]],
    first: bool = False,
    threshold: Union[float, List[float]] = 0.5,
) -> Callable:
    """
    Factory function to create a callable for extracting and discretizing data
    from `ignite.engine.state.output`.

    This function first extracts data specified by the keys from a dictionary or a list of dictionaries,
    then discretizes the extracted data using specified threshold values. If the input data is a list of
    dictionaries and `first` is True, only the first dictionary is considered for extraction.

    Args:
        keys (Union[str, List[str]]): Keys to extract data from the input dictionary or list of dictionaries.
        first (bool): Whether to only extract data from the first dictionary if the input is a list of dictionaries.
        threshold (Union[float, List[float]]): Threshold value(s) for discretization, one for each key.
                                              If a single float is provided, it will be applied to all keys.

    Returns:
        Callable: A function that takes data and returns discretized values for each key. If there is only one key,
                  the returned value will be directly that value, not a tuple containing a single element.
    """
    _keys = ensure_tuple(keys)
    _from_engine_func = from_engine(keys=_keys, first=first)
    # Ensuring that the threshold list is of the same length as keys
    _threshold = ensure_tuple_rep(threshold, len(_keys))

    def _wrapper(data):
        extracted_data = _from_engine_func(data)
        # Ensure the extracted data is always a tuple for consistency
        if not isinstance(extracted_data, tuple):
            extracted_data = (extracted_data,)

        discretized_data = tuple(
            [AsDiscrete(threshold=thr)(arr) for arr, thr in zip(lst, _threshold)]
            for lst in extracted_data
        )
        # If the length of discretized_data is 1, return the first element to avoid returning a list
        return discretized_data if len(discretized_data) > 1 else discretized_data[0]

    return _wrapper


def meta_data_batch_transform(batch):
    """
    Takes in batch from engine.state and returns case name from meta dict
    for the BraTs dataset
    """
    paths = [e["image"].meta[Key.FILENAME_OR_OBJ] for e in batch]
    names = [
        {Key.FILENAME_OR_OBJ: "_".join(path.split("/")[-1].split("_")[:2])}
        for path in paths
    ]
    return names


def meta_data_batch_transform_dir(batch):
    """
    Takes in batch from engine.state and returns case name from meta dict
    for the BraTs dataset
    """
    paths = [e["image"].meta[Key.FILENAME_OR_OBJ] for e in batch]
    names = [{Key.FILENAME_OR_OBJ: path.split("/")[-2]} for path in paths]
    return names


def meta_data_image_transform(images):
    """
    Takes in images from engine.state and returns case name from meta dict
    for datasets with the case name in the filename
    """
    paths = [i.meta[Key.FILENAME_OR_OBJ] for i in images]
    names = ["_".join(path.split("/")[-1].split("_")[:2]) for path in paths]
    return names


def meta_data_image_transform_dir(images):
    """
    Takes in images from engine.state and returns case name from meta dict
    for datasets with the case name in the directory name. eg. Kits23
    """
    paths = [i.meta[Key.FILENAME_OR_OBJ] for i in images]
    names = [path.split("/")[-2] for path in paths]
    return names


class TemperatureScaling(nn.Module):
    """
    A class to wrap a neural network with temperature scaling.
    Output of network needs to be "raw" logits, not probabilities.
    """

    def __init__(
        self,
        network: nn.Module,
        network_ckpt_path: str | None = None,
    ):
        super(TemperatureScaling, self).__init__()
        # load network
        self.network = network
        if network_ckpt_path is not None:
            self.network.load_state_dict(torch.load(network_ckpt_path))
            self.network.eval()  # set to eval mode as we don't want to train the network
        device = next(self.network.parameters()).device
        self.temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)

    def forward(self, input):
        logits = self.network(input)
        return logits / self.temperature

    def parameters(self, recurse: bool = True):
        # Yield only the temperature parameter
        yield self.temperature
