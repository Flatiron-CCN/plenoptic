#!/usr/bin/env python3

import torch
from torch import Tensor
import einops
import warnings
from ...tools.conv import blur_downsample
from typing import Literal


class SimpleAverage(torch.nn.Module):
    """Module to average over the last two dimensions of input"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        """Average over the last two dimensions of input."""
        return torch.mean(x, dim=(-2, -1))


class WeightedAverage(torch.nn.Module):
    """Module to take a weighted average over last two dimensions of tensors.

    - Weights are set at initialization, must be non-negative, and 3d Tensors (different
      weighting regions indexed on first dimension, height and width on last two).

    - If two weights are set, they are multiplied together when taking the average
      (e.g., separable polar angle and eccentricity weights).

    - Weights are normalized at initialization so that they sum to 1 (as best as
      possible). If any weighting region sums to near-zero, an exception is raised. If
      there's variation across weighting region sums, a warning is raised.

    Parameters
    ----------
    weights_1, weights_2 :
        3d Tensors defining the weights for the average.
    image_shape :
        Last two dimensions of weights tensors

    Attributes
    ----------
    weights :
        List of one or two 3d Tensors (depending on whether weights_2 set at
        initialization) containing the normalized weights.

    """

    def __init__(self, weights_1: Tensor, weights_2: Tensor | None = None):
        super().__init__()
        self._validate_weights(weights_1, "_1")
        self.register_buffer("_weights_1", weights_1)
        self._n_weights = 1
        input_einsum = "w1 h w"
        output_einsum = "w1"
        if weights_2 is not None:
            self._validate_weights(weights_2, "_2")
            if weights_1.shape[-2:] != weights_2.shape[-2:]:
                raise ValueError(
                    "weights_1 and weights_2 must have same height and width!"
                )
            self._n_weights += 1
            input_einsum += ", w2 h w"
            output_einsum += " w2"
            self.register_buffer("_weights_2", weights_2)
        self.image_shape = weights_1.shape[-2:]
        self._input_einsum = input_einsum
        self._output_einsum = output_einsum
        self._weight_einsum = f"{input_einsum} -> {output_einsum}"
        self._forward_einsum = f"{input_einsum}, b c {{extra_dims}} h w -> b c {output_einsum} {{extra_dims}}"
        self._extra_dims = ["", "i1", "i1 i2"]
        self._normalize_weights()

    @property
    def weights(self):
        weights = [self._weights_1]
        if self._n_weights > 1:
            weights.append(self._weights_2)
        return weights

    def _normalize_weights(self):
        """Normalize weights.

        Call sum_weights() to multiply and sum all weights, then:

        - Check whether any weighting region sum is near-zero. If so, raise ValueError

        - Check variance of weighting region sums and raise warning if that variance is
          not near-zero. (Ideally, all weighting region sums would be the same value.)

        - Take the modal weighting region sum value and divide all weighting regions by
          that value to normalize them. If we have two weight tensors, divide each by
          the sqrt of the mode instead.

        """
        weight_sums = self.sum_weights()
        if torch.isclose(weight_sums, torch.zeros_like(weight_sums)).any():
            raise ValueError(
                "Some of the weights sum to zero! This will not work out well."
            )
        var = weight_sums.var()
        if not torch.isclose(var, torch.zeros_like(var)):
            warnings.warn(
                "Looks like there's some variation in the sums across your weights."
                " That might be fine, but just wanted to make sure you knew..."
            )
        mode = torch.mode(weight_sums.flatten()).values
        if not torch.isclose(mode, torch.ones_like(mode)):
            warnings.warn("Weights don't sum to 1, normalizing...")
            if self._n_weights == 1:
                self._weights_1 = self._weights_1 / mode
            else:
                self._weights_1 = self._weights_1 / mode.sqrt()
                self._weights_2 = self._weights_2 / mode.sqrt()

    @staticmethod
    def _validate_weights(weights: Tensor, idx: "str" = "_1"):
        if weights.ndim != 3:
            raise ValueError(f"weights{idx} must be 3d!")
        if weights.min() < 0:
            raise ValueError(f"weights{idx} must be non-negative!")

    def forward(self, image: Tensor) -> Tensor:
        """Take the weighted average over last two dimensions of input.

        All other dimensions are preserved.

        Parameters
        ----------
        image :
            4d to 6d Tensor.

        Returns
        -------
        weighted_avg :
            Weighted average. Dimensionality depends on both the input's dimensionality
            and ``len(weights)``

        """
        if image.ndim < 4:
            raise ValueError("image must be a tensor of 4 to 6 dimensions!")
        try:
            extra_dims = self._extra_dims[image.ndim - 4]
        except IndexError:
            raise ValueError("image must be a tensor of 4 to 6 dimensions!")
        einsum_str = self._forward_einsum.format(extra_dims=extra_dims)
        return einops.einsum(*self.weights, image, einsum_str).flatten(2, 3)

    def einsum(self, einsum_str: str, *tensors: Tensor) -> Tensor:
        """More general version of forward.

        This takes the input einsum_str and prepends self.weights to it and inserts the
        weight dimensions into the output after "b c" (for batch, channel). Thus this
        will be weird if there's no "b c" dimensions.

        Parameters
        ----------
        einsum_str :
            String of einsum notation, which must contain "b c" in the output. Intended
            use is that this string produces a single output tensor.
        tensors :
            Any number of tensors

        Returns
        -------
        output :
            The result of this einsum

        """
        einsum_str = f"{self._input_einsum}, {einsum_str.split('->')[0]} -> b c {self._output_einsum} {einsum_str.split('->')[1].replace('b c', '')}"
        return einops.einsum(*self.weights, *tensors, einsum_str).flatten(2, 3)

    def sum_weights(self) -> Tensor:
        """Sum weights, largely used for diagnostic purposes.

        Returns
        -------
        sum :
            1d or 2d tensor (depending on ``len(weights)``) containing the sum of all
            weights.

        """
        return einops.einsum(*self.weights, self._weight_einsum)


class WeightedAveragePyramid(torch.nn.Module):
    """Module to take weighted average across scales.

    This initializes a ``WeightedAverage`` per scale, down-sampling by a factor of 2
    using the ``blur_downsample`` method (and normalizing independently).

    As with ``WeightedAverage``:

    - Weights are set at initialization, must be non-negative, and 3d Tensors (different
      weighting regions indexed on first dimension, height and width on last two).

    - If two weights are set, they are multiplied together when taking the average
      (e.g., separable polar angle and eccentricity weights).

    - Weights are normalized at initialization so that they sum to 1 (as best as
      possible). If any weighting region sums to near-zero, an exception is raised. If
      there's variation across weighting region sums, a warning is raised.

    Parameters
    ----------
    weights_1, weights_2 :
        3d Tensors defining the weights for the average.
    n_scales :
        Number of scales.

    Attributes
    ----------
    weights :
        ModuleList of ``WeightedAverage`` at each scale

    """

    def __init__(
        self, weights_1: Tensor, weights_2: Tensor | None = None, n_scales: int = 4
    ):
        super().__init__()
        self._n_weights = 1 if weights_2 is None else 2
        self.n_scales = n_scales
        weights = []
        for i in range(n_scales):
            if i != 0:
                weights_1 = blur_downsample(
                    weights_1.unsqueeze(0), 1, scale_filter=True
                )
                weights_1 = weights_1.squeeze(0).clip(min=0)
                if weights_2 is not None:
                    weights_2 = blur_downsample(
                        weights_2.unsqueeze(0), 1, scale_filter=True
                    )
                    weights_2 = weights_2.squeeze(0).clip(min=0)
            # it's possible negative values will get introduced by the downsampling
            # above, in which case we remove them
            weights.append(WeightedAverage(weights_1, weights_2))
        self.weights = torch.nn.ModuleList(weights)

    def __getitem__(self, idx: int):
        return self.weights[idx]

    def forward(self, image: list[Tensor]) -> list[Tensor]:
        """Take the weighted average over last two dimensions of each input in list.

        All other dimensions are preserved.

        Parameters
        ----------
        image :
            List of 4d to 6d Tensor, each of which has been downsampled by a factor of 2.

        Returns
        -------
        weighted_avg :
            Weighted average. Dimensionality depends on both the input's dimensionality
            and whether ``weights_2`` was set at initialization. Scales are stacked
            along last dimension.

        """
        return torch.stack([w(x) for x, w in zip(image, self.weights)], dim=-1)

    def einsum(self, einsum_str: str, *tensors: list[Tensor]) -> list[Tensor]:
        """More general version of forward, operates on each

        This takes the input einsum_str and prepends self.weights to it and inserts the
        weight dimensions into the output after "b c" (for batch, channel). Thus this
        will be weird if there's no "b c" dimensions.

        Parameters
        ----------
        einsum_str :
            String of einsum notation, which must contain "b c" in the output. Intended
            use is that this string produces a single output tensor.
        tensors :
            Any number of lists of tensors (should all have same number of elements,
            each corresponding to a different scale and thus downsampled by factor of 2).

        Returns
        -------
        output :
            The result of this einsum. Scales are stacked along last dimension.

        """
        return torch.stack(
            [w.einsum(einsum_str, *x) for *x, w in zip(*tensors, self.weights)], dim=-1
        )

    def sum_weights(self) -> Tensor:
        """Sum weights, largely used for diagnostic purposes.

        Returns
        -------
        sum :
            2d or 3d tensor (depending on whether ``weights_2`` was set at
            initialization) containing the sum of all weights on each scale.

        """
        sums = []
        for w in self.weights:
            sums.append(w.sum_weights())
        return einops.pack(sums, f"* {self.weights[0]._output_einsum}")[0]
