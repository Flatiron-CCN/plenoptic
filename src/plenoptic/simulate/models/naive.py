import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from ...tools.conv import same_padding
from ..canonical_computations.filters import circular_gaussian2d

__all__ = ["Identity", "Linear", "Gaussian", "CenterSurround"]


class Identity(torch.nn.Module):
    r"""simple class that just returns a copy of the image

    We use this as a "dummy model" for metrics that we don't have the
    representation for. We use this as the model and then just change
    the objective function.
    """

    def __init__(self, name=None):
        super().__init__()
        if name is not None:
            self.name = name

    def forward(self, img):
        """Return a copy of the image

        Parameters
        ----------
        img : torch.Tensor
            The image to return

        Returns
        -------
        img : torch.Tensor
            a clone of the input image

        """
        y = 1 * img
        return y


class Linear(nn.Module):
    r"""Simplistic linear convolutional model:
    It splits the input greyscale image into low and high frequencies.

    Parameters
    ----------
    kernel_size:
        Convolutional kernel size.
    pad_mode:
        Mode with which to pad image using `nn.functional.pad()`.
    default_filters:
        Initialize the filters to a low-pass and a band-pass.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = (3, 3),
        pad_mode: str = "circular",
        default_filters: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

        self.conv = nn.Conv2d(1, 2, kernel_size, bias=False)

        if default_filters:
            var = torch.as_tensor(3.0)
            f1 = circular_gaussian2d(kernel_size, std=torch.sqrt(var))

            f2 = circular_gaussian2d(kernel_size, std=torch.sqrt(var / 3))

            f2 = f2 - f1
            f2 = f2 / f2.sum()

            self.conv.weight.data = torch.cat([f1, f2], dim=0)

    def forward(self, x: Tensor) -> Tensor:
        y = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        h = self.conv(y)
        return h


class Gaussian(nn.Module):
    """Isotropic Gaussian convolutional filter.
    Kernel elements are normalized and sum to one.

    Parameters
    ----------
    kernel_size:
        Size of convolutional kernel.
    std:
        Standard deviation of circularly symmtric Gaussian kernel.
    pad_mode:
        Padding mode argument to pass to `torch.nn.functional.pad`.
    out_channels:
        Number of filters with which to convolve.
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass. Cached to `self._filt`.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        std: float | Tensor = 3.0,
        pad_mode: str = "reflect",
        out_channels: int = 1,
        cache_filt: bool = False,
    ):
        super().__init__()
        assert std > 0, "Gaussian standard deviation must be positive"
        if isinstance(std, float) or std.shape == torch.Size([]):
            std = torch.ones(out_channels) * std
        self.std = nn.Parameter(torch.as_tensor(std))

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode
        self.out_channels = out_channels

        self.cache_filt = cache_filt
        self.register_buffer("_filt", None)

    @property
    def filt(self):
        if self._filt is not None:  # use old filter
            return self._filt
        else:  # create new filter, optionally cache it
            filt = circular_gaussian2d(self.kernel_size, self.std, self.out_channels)

            if self.cache_filt:
                self.register_buffer("_filt", filt)
            return filt

    def forward(self, x: Tensor, **conv2d_kwargs) -> Tensor:
        self.std.data = self.std.data.abs()  # ensure stdev is positive

        x = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)
        y = F.conv2d(x, self.filt, **conv2d_kwargs)

        return y


class CenterSurround(nn.Module):
    """Center-Surround, Difference of Gaussians (DoG) filter model. Can be either
    on-center/off-surround, or vice versa.

    Filter is constructed as:
    .. math::
        f &= amplitude_ratio * center - surround \\
        f &= f/f.sum()

    The signs of center and surround are determined by ``on_center`` argument.

    Parameters
    ----------
    kernel_size:
        Shape of convolutional kernel.
    on_center:
        Dictates whether center is on or off; surround will be the opposite of center
        (i.e. on-off or off-on). If List of bools, then list length must equal
        `out_channels`, if just a single bool, then all `out_channels` will be assumed
        to be all on-off or off-on.
    amplitude_ratio:
        Ratio of center/surround amplitude. Applied before filter normalization.
    center_std:
        Standard deviation of circular Gaussian for center.
    surround_std:
        Standard deviation of circular Gaussian for surround. Must be at least
        `ratio_limit` times `center_std`.
    out_channels:
        Number of filters.
    pad_mode:
        Padding for convolution, defaults to "circular".
    cache_filt:
        Whether or not to cache the filter. Avoids regenerating filt with each
        forward pass. Cached to `self._filt`
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        on_center: bool | list[bool] = True,
        amplitude_ratio: float = 1.25,
        center_std: float | Tensor = 1.0,
        surround_std: float | Tensor = 4.0,
        out_channels: int = 1,
        pad_mode: str = "reflect",
        cache_filt: bool = False,
    ):
        super().__init__()

        # make sure each channel is on-off or off-on
        if isinstance(on_center, bool):
            on_center = [on_center] * out_channels
        assert len(on_center) == out_channels, "len(on_center) must match out_channels"

        # make sure each channel has a center and surround std
        if isinstance(center_std, float) or center_std.shape == torch.Size([]):
            center_std = torch.ones(out_channels) * center_std
        if isinstance(surround_std, float) or surround_std.shape == torch.Size([]):
            surround_std = torch.ones(out_channels) * surround_std
        assert len(center_std) == out_channels and len(surround_std) == out_channels, (
            "stds must correspond to each out_channel"
        )
        assert amplitude_ratio >= 1.0, "ratio of amplitudes must at least be 1."

        self.on_center = on_center

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.register_buffer("amplitude_ratio", torch.as_tensor(amplitude_ratio))

        self.center_std = nn.Parameter(torch.ones(out_channels) * center_std)
        self.surround_std = nn.Parameter(torch.ones(out_channels) * surround_std)

        self.out_channels = out_channels
        self.pad_mode = pad_mode

        self.cache_filt = cache_filt
        self.register_buffer("_filt", None)

    @property
    def filt(self) -> Tensor:
        """Creates an on center/off surround, or off center/on surround conv filter"""
        if self._filt is not None:  # use cached filt
            return self._filt
        else:  # generate new filt and optionally cache
            on_amp = self.amplitude_ratio
            device = on_amp.device

            filt_center = circular_gaussian2d(
                self.kernel_size, self.center_std, self.out_channels
            )
            filt_surround = circular_gaussian2d(
                self.kernel_size, self.surround_std, self.out_channels
            )

            # sign is + or - depending on center is on or off
            sign = torch.as_tensor([1.0 if x else -1.0 for x in self.on_center]).to(
                device
            )
            sign = sign.view(self.out_channels, 1, 1, 1)

            filt = on_amp * (sign * (filt_center - filt_surround))

            if self.cache_filt:
                self.register_buffer("_filt", filt)
        return filt

    def forward(self, x: Tensor) -> Tensor:
        x = same_padding(x, self.kernel_size, pad_mode=self.pad_mode)

        y = F.conv2d(x, self.filt, bias=None)
        return y
