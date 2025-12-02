from typing import Callable, Optional, List, Tuple, Union

import torch
from torch import nn, optim
from torch.nn import functional as F

from monai.handlers.utils import from_engine
from monai.utils.misc import ensure_tuple, ensure_tuple_rep, ImageMetaKey as Key
from monai.config.type_definitions import KeysCollection

from monai.transforms.post.array import AsDiscrete

TensorOrList = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]

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
    threshold: Union[float, List[float], None] = None,
    argmax: bool = False,
    to_onehot: Optional[int] = None,
    rounding: Optional[str] = None,
    **kwargs,
) -> Callable:
    """
    Factory function to create a callable for extracting and discretizing data
    from `ignite.engine.state.output`.

    This function first extracts data specified by the keys from a dictionary or a list of dictionaries,
    then discretizes the extracted data using AsDiscrete transform with specified parameters. If the input
    data is a list of dictionaries and `first` is True, only the first dictionary is considered for extraction.

    Args:
        keys (Union[str, List[str]]): Keys to extract data from the input dictionary or list of dictionaries.
        first (bool): Whether to only extract data from the first dictionary if the input is a list of dictionaries.
        threshold (Union[float, List[float], None]): Threshold value(s) for discretization, one for each key.
                                                    If a single float is provided, it will be applied to all keys.
                                                    If None, no thresholding is applied.
        argmax (bool): Whether to execute argmax function on input data before transform.
        to_onehot (Optional[int]): If not None, convert input data into the one-hot format with specified
                                  number of classes.
        rounding (Optional[str]): If not None, round the data according to the specified option,
                                 available options: ["torchrounding"].
        **kwargs: Additional parameters to torch.argmax, monai.networks.one_hot.
                 Currently dim, keepdim, dtype are supported, unrecognized parameters will be ignored.
                 These default to 0, True, torch.float respectively.

    Returns:
        Callable: A function that takes data and returns discretized values for each key. If there is only one key,
                  the returned value will be directly that value, not a tuple containing a single element.
    """
    _keys = ensure_tuple(keys)
    _from_engine_func = from_engine(keys=_keys, first=first)

    # Handle threshold parameter - ensure it's either None or properly replicated
    if threshold is not None:
        _threshold = ensure_tuple_rep(threshold, len(_keys))
    else:
        _threshold = [None] * len(_keys)

    def _wrapper(data):
        extracted_data = _from_engine_func(data)
        # Ensure the extracted data is always a tuple for consistency
        if not isinstance(extracted_data, tuple):
            extracted_data = (extracted_data,)

        discretized_data = []
        for batch_data in extracted_data:
            # batch_data is a list of tensors for each item in the batch
            batch_discretized = []
            for arr, thr in zip(batch_data, _threshold):
                discretized = AsDiscrete(
                    threshold=thr,
                    argmax=argmax,
                    to_onehot=to_onehot,
                    rounding=rounding,
                    **kwargs,
                )(arr)
                batch_discretized.append(discretized)
            discretized_data.append(batch_discretized)

        discretized_data = tuple(discretized_data)
        # If the length of discretized_data is 1, return the first element to avoid returning a tuple with single element
        return discretized_data[0] if len(discretized_data) == 1 else discretized_data

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


# class TemperatureScaling(nn.Module):
#     """
#     Wrap a trained segmentation network with temperature scaling.
#     The wrapped network must output raw (unnormalized) logits.

#     By default, if the network returns deep-supervision outputs (list/tuple),
#     we use only the first (highest-resolution) output for calibration/inference.
#     """

#     def __init__(
#         self,
#         network: nn.Module,
#         network_ckpt_path: Optional[str] = None,
#         use_main_output_only: bool = True,
#     ):
#         super().__init__()
#         self.network = network

#         if network_ckpt_path is not None:
#             checkpoint = torch.load(network_ckpt_path, map_location="cpu")
#             self.network.load_state_dict(checkpoint)

#         self.network.eval()
#         for p in self.network.parameters():
#             p.requires_grad_(False)

#         # Unconstrained parameter; map -> positive T with softplus
#         self._log_t = nn.Parameter(torch.zeros(()))
#         self.use_main_output_only = use_main_output_only

#     @property
#     def temperature(self) -> torch.Tensor:
#         # softplus for stable positivity
#         return F.softplus(self._log_t) + 1e-6

#     def _scale(self, logits: torch.Tensor) -> torch.Tensor:
#         return logits / self.temperature

#     def forward(self, x: torch.Tensor) -> TensorOrList:
#         logits = self.network(x)

#         if isinstance(logits, (list, tuple)):
#             if self.use_main_output_only:
#                 # Calibrate only the main head (typically logits[0])
#                 return self._scale(logits[0])
#             else:
#                 # Maintain structure; scale every head (usually unnecessary)
#                 return [self._scale(o) for o in logits]
#         else:
#             return self._scale(logits)

#     def get_temperature(self) -> float:
#         return float(self.temperature.item())


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
        if isinstance(logits, (list, tuple)):
            return logits[0] / self.temperature
        else:
            return logits / self.temperature

    def parameters(self, recurse: bool = True):
        # Yield only the temperature parameter
        yield self.temperature
