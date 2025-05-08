import torch
from monai.transforms import MapTransform

__all__ = ["ConvertToKits23Classesd", "ConvertToKits23ClassesSoftmaxd"]

# Hierarchical Evaluation Classes


class ConvertToKits23Classesd(MapTransform):
    """
    Convert class labels [0, 1, 2, 3] to "one-hot" encoded format for KiTS23 segmentation task.
    Works with unbatched data with shape [1, H, W, D].

    IN (As stored on disk):
    0: Background
    1: Kidney
    2: Tumor
    3: Cyst

    OUT ONEHOT (As required for eval):
    1: Tumor = Tumor (2)
    2: Kidney Mass = Tumor (2) + Cyst (3)
    3: Kidney and Masses = Kidney (1) + Tumor (2) + Cyst (3)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            x = d[key]  # x is expected to have the shape [1, H, W, D]

            result = []
            # 0 - Background (BG)
            result.append((x == 0).float().squeeze(0))  # Remove the singleton dimension
            # 1 - Tumor (T)
            result.append((x == 2).float().squeeze(0))
            # 2 - Kidney Mass (KM) = Tumor (2) + Cyst (3)
            km = (torch.logical_or(x == 2, x == 3)).float().squeeze(0)
            result.append(km)
            # 3 - Kidney and Masses (K&M) = Kidney (1) + Tumor (2) + Cyst (3)
            k_m = (
                (torch.logical_or(torch.logical_or(x == 1, x == 2), x == 3))
                .float()
                .squeeze(0)
            )
            result.append(k_m)

            # Stack along the channel dimension to get the shape [C, H, W, D]
            d[key] = torch.stack(result, dim=0)
        return d


class ConvertToKits23ClassesSoftmaxd(MapTransform):
    """
    This transformation converts softmax class probabilities into combined KiTS23 tumor
    classes. The transformation is designed for use with unbatched data, where each
    item has the shape [C, H, W, D] and C represents the number of classes.

    The data[key] can be either a tensor of [C, H, W, D] or a list of such tensors,
    as a list is what is resultant from a network with deep supervision such as SegResNet

    IN: Multichannel softmax values
    0: Background
    1: Kidney
    2: Tumor
    3: Cyst

    OUT (As required for eval):
    1: Tumor = Tumor (2)
    2: Kidney Mass = Tumor (2) + Cyst (3)
    3: Kidney and Masses = Kidney (1) + Tumor (2) + Cyst (3)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            probabilities = d[key]

            # Check if probabilities is a list of tensors
            if isinstance(probabilities, list):
                d[key] = [self.convert(prob) for prob in probabilities]
            else:
                d[key] = self.convert(probabilities)

        return d

    def convert(self, probabilities):
        result = []
        # 0 - Background (BG)
        result.append(probabilities[0, ...])
        # 1 - Tumor (T)
        result.append(probabilities[2, ...])
        # 2 - Kidney Mass (KM)
        km = probabilities[2, ...] + probabilities[3, ...]
        result.append(km)
        # 3 - Kidney and Masses (K&M)
        k_m = probabilities[1, ...] + probabilities[2, ...] + probabilities[3, ...]
        result.append(k_m)

        # Stack along the channel dimension to maintain the shape [C, H, W, D]
        return torch.stack(result, dim=0)
