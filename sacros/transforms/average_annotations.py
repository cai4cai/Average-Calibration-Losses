import torch
from monai.transforms import MapTransform, RandomizableTransform
from typing import Any, Dict, Hashable, Mapping, Sequence

__all__ = ["AverageAnnotationsd", "RandomWeightedAverageAnnotationsd"]


class AverageAnnotationsd(MapTransform):
    """
    Dictionary-based transform to average multiple label annotations into a single label tensor.

    Args:
        keys: List of keys to fetch the label annotations from the input dictionary.
        output_key: Key to store the averaged label annotation in the output dictionary.
        allow_missing_keys: If True, will not raise an exception if a key is missing.
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        output_key: Hashable,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """
        Apply the transform to the data.

        Args:
            data: Dictionary containing the data to be transformed.

        Returns:
            Dictionary with the averaged label annotation added.
        """
        annotations = torch.stack([data[key] for key in self.keys], dim=0)
        averaged_annotation = torch.mean(annotations, dim=0)
        data[self.output_key] = averaged_annotation
        for key in self.keys:
            del data[key]
        return data


class RandomWeightedAverageAnnotationsd(RandomizableTransform, MapTransform):
    """
    Dictionary-based transform to create a weighted average of multiple label annotations,
    using random weights for the combination.

    Args:
        keys: List of keys to fetch the label annotations from the input dictionary.
        output_key: Key to store the weighted average label annotation in the output dictionary.
        prob: Probability of applying the transform.
        allow_missing_keys: If True, will not raise an exception if a key is missing.
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        output_key: Hashable,
        prob: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.output_key = output_key
        self.weights = None

    def randomize(self, data: Any = None) -> None:
        """Generate random weights for averaging the annotations."""
        self.weights = self.R.rand(len(self.keys))
        self.weights /= self.weights.sum()

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        """
        Apply the transform to the data.

        Args:
            data: Dictionary containing the data to be transformed.

        Returns:
            Dictionary with the weighted average label annotation added.
        """
        self.randomize()
        annotations = torch.stack([data[key] for key in self.keys], dim=0)
        weighted_annotation = torch.sum(
            annotations * torch.tensor(self.weights).view(-1, 1, 1, 1, 1), dim=0
        )
        data[self.output_key] = weighted_annotation
        for key in self.keys:
            del data[key]
        return data
