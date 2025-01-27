"""Steerable frequency pyramid

Construct a steerable pyramid on matrix two dimensional signals, in the
Fourier domain.
"""

import warnings
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from einops import rearrange
from numpy.typing import NDArray
from scipy.special import factorial
from torch import Tensor

from ...tools.signal import interpolate1d, raised_cosine, steer

complex_types = [torch.cdouble, torch.cfloat]

SCALES_TYPE = int | Literal["residual_lowpass", "residual_highpass"]
KEYS_TYPE = tuple[int, int] | Literal["residual_lowpass", "residual_highpass"]


class SteerablePyramidFreq(nn.Module):
    r"""Steerable frequency pyramid in Torch

    Construct a steerable pyramid on matrix two dimensional signals, in the
    Fourier domain. Boundary-handling is circular. Reconstruction is exact
    (within floating point errors). However, if the image has an odd-shape,
    the reconstruction will not be exact due to boundary-handling issues
    that have not been resolved.

    The squared radial functions tile the Fourier plane with a raised-cosine
    falloff. Angular functions are cos(theta-k*pi/order+1)^(order).

    Notes
    -----
    Transform described in [1]_, filter kernel design described in [2]_.
    For further information see the project webpage_

    Parameters
    ----------
    image_shape : `list or tuple`
        shape of input image
    height : 'auto' or `int`
        The height of the pyramid. If 'auto', will automatically determine based on the
        size of `image`. If an int, must be non-negative and less than
        log2(min(image_shape[1], image_shape[1]))-2. If height=0, this only returns the
        residuals.
    order : `int`.
        The Gaussian derivative order used for the steerable filters, in [1,
        15]. Note that to achieve steerability the minimum number of
        orientation is `order` + 1, and is used here. To get more orientations
        at the same order, use the method `steer_coeffs`
    twidth : `int`
        The width of the transition region of the radial lowpass function, in
        octaves
    is_complex : `bool`
        Whether the pyramid coefficients should be complex or not. If True, the
        real and imaginary parts correspond to a pair of even and odd symmetric
        filters. If False, the coefficients only include the real part / even
    downsample: `bool`
        Whether to downsample each scale in the pyramid or keep the output
        pyramid coefficients in fixed bands of size imshapeximshape. When
        downsample is False, the forward method returns a tensor.
    tight_frame: `bool` default: False
        Whether the pyramid obeys the generalized parseval theorem or not (i.e.
        is a tight frame). If True, the energy of the pyr_coeffs = energy of
        the image. If not this is not true. In order to match the
        matlabPyrTools or pyrtools pyramids, this must be set to False

    Attributes
    ----------
    image_shape : `list or tuple`
        shape of input image
    pyr_size : `dict`
        Dictionary containing the sizes of the pyramid coefficients. Keys are
        `(level, band)` tuples and values are tuples.
    fft_norm : `str`
        The way the ffts are normalized, see pytorch documentation for more details.
    is_complex : `bool`
        Whether the coefficients are complex- or real-valued.

    References
    ----------
    .. [1] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible
       Architecture for Multi-Scale Derivative Computation," Second Int'l Conf
       on Image Processing, Washington, DC, Oct 1995.
    .. [2] A Karasaridis and E P Simoncelli, "A Filter Design Technique for
       Steerable Pyramid Image Transforms", ICASSP, Atlanta, GA, May 1996. ..
       _webpage: https://www.cns.nyu.edu/~eero/steerpyr/

    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        height: Literal["auto"] | int = "auto",
        order: int = 3,
        twidth: int = 1,
        is_complex: bool = False,
        downsample: bool = True,
        tight_frame: bool = False,
    ):
        super().__init__()

        self.pyr_size = OrderedDict()
        self.order = order
        self.image_shape = image_shape

        if (self.image_shape[0] % 2 != 0) or (self.image_shape[1] % 2 != 0):
            warnings.warn("Reconstruction will not be perfect with odd-sized images")

        self.is_complex = is_complex
        self.downsample = downsample
        self.tight_frame = tight_frame
        if self.tight_frame:
            self.fft_norm = "ortho"
        else:
            self.fft_norm = "backward"
        # cache constants
        self.lutsize = 1024
        self.Xcosn = (
            np.pi
            * np.array(range(-(2 * self.lutsize + 1), (self.lutsize + 2)))
            / self.lutsize
        )
        self.alpha = (self.Xcosn + np.pi) % (2 * np.pi) - np.pi

        max_ht = np.floor(np.log2(min(self.image_shape[0], self.image_shape[1]))) - 2
        if height == "auto":
            self.num_scales = int(max_ht)
        elif height > max_ht:
            raise ValueError(f"Cannot build pyramid higher than {max_ht:.0f} levels.")
        elif height < 0:
            raise ValueError("Height must be a non-negative integer.")
        else:
            self.num_scales = int(height)

        if self.order > 15 or self.order <= 0:
            raise ValueError("order must be an integer in the range [1,15].")
        self.num_orientations = int(self.order + 1)

        if twidth <= 0:
            raise ValueError("twidth must be positive.")
        twidth = int(twidth)

        dims = np.array(self.image_shape)

        # make a grid for the raised cosine interpolation
        ctr = np.ceil((np.array(dims) + 0.5) / 2).astype(int)

        (xramp, yramp) = np.meshgrid(
            np.linspace(-1, 1, dims[1] + 1)[:-1],
            np.linspace(-1, 1, dims[0] + 1)[:-1],
        )

        self.angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0] - 1, ctr[1] - 1] = log_rad[ctr[0] - 1, ctr[1] - 2]
        self.log_rad = np.log2(log_rad)

        # radial transition function (a raised cosine in log-frequency):
        self.Xrcos, Yrcos = raised_cosine(twidth, (-twidth / 2.0), np.array([0, 1]))
        self.Yrcos = np.sqrt(Yrcos)

        self.YIrcos = np.sqrt(1.0 - self.Yrcos**2)

        # create low and high masks
        lo0mask = interpolate1d(self.log_rad, self.YIrcos, self.Xrcos)
        hi0mask = interpolate1d(self.log_rad, self.Yrcos, self.Xrcos)
        self.register_buffer("lo0mask", torch.as_tensor(lo0mask).unsqueeze(0))
        self.register_buffer("hi0mask", torch.as_tensor(hi0mask).unsqueeze(0))

        # need a mock image to down-sample so that we correctly
        # construct the differently-sized masks
        mock_image = np.random.rand(*self.image_shape)
        imdft = np.fft.fftshift(np.fft.fft2(mock_image))
        lodft = imdft * lo0mask

        # this list, used by coarse-to-fine optimization, gives all the
        # scales (including residuals) from coarse to fine
        self.scales = (
            ["residual_lowpass"]
            + list(range(self.num_scales))[::-1]
            + ["residual_highpass"]
        )

        # we create these copies because they will be modified in the
        # following loops
        Xrcos = self.Xrcos.copy()
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        # pre-generate the angle, hi and lo masks, as well as the
        # indices used for down-sampling.
        self._loindices = []
        for i in range(self.num_scales):
            Xrcos -= np.log2(2)
            const = (
                (2 ** (2 * self.order))
                * (factorial(self.order, exact=True) ** 2)
                / float(self.num_orientations * factorial(2 * self.order, exact=True))
            )

            if self.is_complex:
                Ycosn_forward = (
                    2.0
                    * np.sqrt(const)
                    * (np.cos(self.Xcosn) ** self.order)
                    * (np.abs(self.alpha) < np.pi / 2.0).astype(int)
                )
                Ycosn_recon = np.sqrt(const) * (np.cos(self.Xcosn)) ** self.order

            else:
                Ycosn_forward = np.sqrt(const) * (np.cos(self.Xcosn)) ** self.order
                Ycosn_recon = Ycosn_forward

            himask = interpolate1d(log_rad, self.Yrcos, Xrcos)
            self.register_buffer(
                f"_himasks_scale_{i}", torch.as_tensor(himask).unsqueeze(0)
            )

            anglemasks = []
            anglemasks_recon = []
            for b in range(self.num_orientations):
                anglemask = interpolate1d(
                    angle,
                    Ycosn_forward,
                    self.Xcosn + np.pi * b / self.num_orientations,
                )
                anglemask_recon = interpolate1d(
                    angle,
                    Ycosn_recon,
                    self.Xcosn + np.pi * b / self.num_orientations,
                )
                anglemasks.append(torch.as_tensor(anglemask).unsqueeze(0))
                anglemasks_recon.append(torch.as_tensor(anglemask_recon).unsqueeze(0))

            self.register_buffer(f"_anglemasks_scale_{i}", torch.cat(anglemasks))
            self.register_buffer(
                f"_anglemasks_recon_scale_{i}", torch.cat(anglemasks_recon)
            )
            if not self.downsample:
                lomask = interpolate1d(log_rad, self.YIrcos, Xrcos)
                self.register_buffer(
                    f"_lomasks_scale_{i}", torch.as_tensor(lomask).unsqueeze(0)
                )
                self._loindices.append([np.array([0, 0]), dims])
                lodft = lodft * lomask

            else:
                # subsample lowpass
                dims = np.array([lodft.shape[0], lodft.shape[1]])
                ctr = np.ceil((dims + 0.5) / 2).astype(int)
                lodims = np.ceil((dims - 0.5) / 2).astype(int)
                loctr = np.ceil((lodims + 0.5) / 2).astype(int)
                lostart = ctr - loctr
                loend = lostart + lodims
                self._loindices.append([lostart, loend])

                # subsample indices
                log_rad = log_rad[lostart[0] : loend[0], lostart[1] : loend[1]]
                angle = angle[lostart[0] : loend[0], lostart[1] : loend[1]]

                lomask = interpolate1d(log_rad, self.YIrcos, Xrcos)
                self.register_buffer(
                    f"_lomasks_scale_{i}", torch.as_tensor(lomask).unsqueeze(0)
                )
                # subsampling
                lodft = lodft[lostart[0] : loend[0], lostart[1] : loend[1]]
                # convolution in spatial domain
                lodft = lodft * lomask

        # reasonable default dtype
        self.to(torch.float32)

    def forward(
        self,
        x: Tensor,
        scales: list[SCALES_TYPE] | None = None,
    ) -> OrderedDict:
        r"""Generate the steerable pyramid coefficients for an image

        Parameters
        ----------
        x :
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width).
        scales :
            Which scales to include in the returned representation. If None, we
            include all scales. Otherwise, can contain subset of values present
            in this model's ``scales`` attribute (ints from 0 up to
            ``self.num_scales-1`` and the strs 'residual_highpass' and
            'residual_lowpass'. Can contain a single value or multiple values.
            If it's an int, we include all orientations from that scale. Order
            within the list does not matter.

        Returns
        -------
        representation:
            Pyramid coefficients

        """
        pyr_coeffs = OrderedDict()
        if scales is None:
            scales = self.scales
        scale_ints = [s for s in scales if isinstance(s, int)]
        if len(scale_ints) != 0:
            assert (max(scale_ints) < self.num_scales) and (min(scale_ints) >= 0), (
                "Scales must be within 0 and num_scales-1"
            )
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        lo0mask = self.lo0mask.clone()
        hi0mask = self.hi0mask.clone()

        # x is a torch tensor batch of images of size (batch, channel, height,
        # width)
        assert len(x.shape) == 4, "Input must be batch of images of shape BxCxHxW"

        imdft = fft.fft2(x, dim=(-2, -1), norm=self.fft_norm)
        imdft = fft.fftshift(imdft)

        if "residual_highpass" in scales:
            # high-pass
            hi0dft = imdft * hi0mask
            hi0 = fft.ifftshift(hi0dft)
            hi0 = fft.ifft2(hi0, dim=(-2, -1), norm=self.fft_norm)
            pyr_coeffs["residual_highpass"] = hi0.real
            self.pyr_size["residual_highpass"] = tuple(hi0.real.shape[-2:])

        # input to the next scale is the low-pass filtered component
        lodft = imdft * lo0mask

        for i in range(self.num_scales):
            if i in scales:
                # high-pass mask is selected based on the current scale
                himask = getattr(self, f"_himasks_scale_{i}")
                # compute filter output at each orientation
                for b in range(self.num_orientations):
                    # band pass filtering is done in the fourier space as multiplying
                    #  by the fft of a gaussian derivative.
                    # The oriented dft is computed as a product of the fft of the
                    # low-passed component, the precomputed anglemask (specifies
                    # orientation), and the precomputed hipass mask (creating a bandpass
                    # filter) the complex_const variable comes from the Fourier
                    # transform of a gaussian derivative.
                    # Based on the order of the gaussian, this constant changes.

                    anglemask = getattr(self, f"_anglemasks_scale_{i}")[b]

                    complex_const = np.power(complex(0, -1), self.order)
                    banddft = complex_const * lodft * anglemask * himask
                    # fft output is then shifted to center frequencies
                    band = fft.ifftshift(banddft)
                    # ifft is applied to recover the filtered representation in spatial
                    # domain
                    band = fft.ifft2(band, dim=(-2, -1), norm=self.fft_norm)

                    # for real pyramid, take the real component of the complex band
                    if not self.is_complex:
                        pyr_coeffs[(i, b)] = band.real
                    else:
                        # Because the input signal is real, to maintain a tight frame
                        # if the complex pyramid is used, magnitudes need to be divided
                        # by sqrt(2) because energy is doubled.

                        if self.tight_frame:
                            band = band / np.sqrt(2)
                        pyr_coeffs[(i, b)] = band
                    self.pyr_size[(i, b)] = tuple(band.shape[-2:])

            if not self.downsample:
                # no subsampling of angle and rad
                # just use lo0mask
                lomask = getattr(self, f"_lomasks_scale_{i}")
                lodft = lodft * lomask

                # Since we don't subsample here, if we are not using
                # orthonormalization that we need to manually account for the
                # subsampling, so that energy in each band remains the same
                # the energy is cut by factor of 4 so we need to scale magnitudes
                # by factor of 2.

                if self.fft_norm != "ortho":
                    lodft = 2 * lodft
            else:
                # subsample indices
                lostart, loend = self._loindices[i]

                log_rad = log_rad[lostart[0] : loend[0], lostart[1] : loend[1]]
                angle = angle[lostart[0] : loend[0], lostart[1] : loend[1]]

                # subsampling of the dft for next scale
                lodft = lodft[:, :, lostart[0] : loend[0], lostart[1] : loend[1]]
                # low-pass filter mask is selected
                lomask = getattr(self, f"_lomasks_scale_{i}")
                # again multiply dft by subsampled mask (convolution in spatial domain)

                lodft = lodft * lomask

        if "residual_lowpass" in scales:
            # compute residual lowpass when height <=1
            lo0 = fft.ifftshift(lodft)
            lo0 = fft.ifft2(lo0, dim=(-2, -1), norm=self.fft_norm)
            pyr_coeffs["residual_lowpass"] = lo0.real
            self.pyr_size["residual_lowpass"] = tuple(lo0.real.shape[-2:])

        return pyr_coeffs

    @staticmethod
    def convert_pyr_to_tensor(
        pyr_coeffs: OrderedDict, split_complex: bool = False
    ) -> tuple[Tensor, tuple[int, bool, list[KEYS_TYPE]]]:
        r"""Convert coefficient dictionary to a tensor.

        The output tensor has shape (batch, channel, height, width) and is
        intended to be used in an ``torch.nn.Module`` downstream. In the
        multichannel case, all bands for each channel will be stacked together
        (i.e. if there are 2 channels and 18 bands per channel,
        pyr_tensor[:,0:18,...] will contain the pyr responses for channel 1 and
        pyr_tensor[:, 18:36, ...] will contain the responses for channel 2). In
        the case of a complex, multichannel pyramid with split_complex=True,
        the real/imaginary bands will be intereleaved so that they appear as
        pairs with neighboring indices in the channel dimension of the tensor
        (Note: the residual bands are always real so they will only ever have a
        single band even when split_complex=True.)

        This only works if ``pyr_coeffs`` was created with a pyramid with
        ``downsample=False``

        Parameters
        ----------
        pyr_coeffs:
            the pyramid coefficients
        split_complex:
            indicates whether the output should split complex bands into
            real/imag channels or keep them as a single channel. This should be
            True if you intend to use a convolutional layer on top of the
            output.

        Returns
        -------
        pyr_tensor:
            shape (batch, channel, height, width). pyramid coefficients
            reshaped into tensor. The first channel will be the residual
            highpass and the last will be the residual lowpass. Each band is
            then a separate channel.
        pyr_info:
            Information required to recreate the dictionary, containing the
            number of channels, if split_complex was used in this function
            call, and the list of pyramid keys for the dictionary

        See also
        --------
        convert_tensor_to_pyr:
            Convert tensor representation to pyramid dictionary.

        """

        pyr_keys = list(pyr_coeffs.keys())
        test_band = pyr_coeffs[pyr_keys[0]]
        num_channels = test_band.size(1)
        coeff_list = []
        for ch in range(num_channels):
            coeff_list_resid = []
            coeff_list_bands = []
            for k in pyr_keys:
                coeffs = pyr_coeffs[k][:, ch : (ch + 1), ...]
                if "residual" in k:
                    coeff_list_resid.append(coeffs)
                else:
                    if (coeffs.dtype in complex_types) and split_complex:
                        coeff_list_bands.extend([coeffs.real, coeffs.imag])
                    else:
                        coeff_list_bands.append(coeffs)

            if "residual_highpass" in pyr_coeffs:
                coeff_list_bands.insert(0, coeff_list_resid[0])
                if "residual_lowpass" in pyr_coeffs:
                    coeff_list_bands.append(coeff_list_resid[1])
            elif "residual_lowpass" in pyr_coeffs:
                coeff_list_bands.append(coeff_list_resid[0])

            coeff_list.extend(coeff_list_bands)

        try:
            pyr_tensor = torch.cat(coeff_list, dim=1)
            pyr_info = tuple([num_channels, split_complex, pyr_keys])
        except RuntimeError:
            raise Exception(
                """feature maps could not be concatenated into tensor. Check that you
            are using coefficients that are not downsampled across scales.
            This is done with the 'downsample=False' argument for the pyramid"""
            )

        return pyr_tensor, pyr_info

    @staticmethod
    def convert_tensor_to_pyr(
        pyr_tensor: Tensor,
        num_channels: int,
        split_complex: bool,
        pyr_keys: list[KEYS_TYPE],
    ) -> OrderedDict:
        r"""Convert pyramid coefficient tensor to dictionary format.

        ``num_channels``, ``split_complex``, and ``pyr_keys`` are elements of
        the ``pyr_info`` tuple returned by ``convert_pyr_to_tensor``. You
        should always unpack the arguments for this function from that
        ``pyr_info`` tuple. Example Usage:

        .. code-block:: python

           pyr_tensor, pyr_info = convert_pyr_to_tensor(pyr_coeffs, split_complex=True)
           pyr_dict = convert_tensor_to_pyr(pyr_tensor, *pyr_info)

        Parameters
        ----------
        pyr_tensor:
            Shape (batch, channel, height, width). The pyramid coefficients
        num_channels:
            number of channels in the original input tensor the pyramid was
            created for (i.e. if the input was an RGB image, this would be 3)
        split_complex:
            true or false, specifying whether the pyr_tensor was created with
            complex channels split or not (if the pyramid was a complex
            pyramid).
        pyr_keys:
            tuple containing the list of keys for the original pyramid dictionary

        Returns
        -------
        pyr_coeffs:
            pyramid coefficients in dictionary format

        See also
        --------
        convert_pyr_to_tensor:
            Convert pyramid dictionary representation to tensor.

        """

        pyr_coeffs = OrderedDict()
        i = 0
        for ch in range(num_channels):
            for k in pyr_keys:
                if "residual" in k:
                    band = pyr_tensor[:, i, ...].unsqueeze(1).type(torch.float)
                    i += 1
                else:
                    if split_complex:
                        band = torch.view_as_complex(
                            rearrange(
                                pyr_tensor[:, i : i + 2, ...],
                                "b c h w -> b h w c",
                            )
                            .unsqueeze(1)
                            .contiguous()
                        )
                        i += 2
                    else:
                        band = pyr_tensor[:, i, ...].unsqueeze(1)
                        i += 1
                if k not in pyr_coeffs:
                    pyr_coeffs[k] = band
                else:
                    pyr_coeffs[k] = torch.cat([pyr_coeffs[k], band], dim=1)

        return pyr_coeffs

    def _recon_levels_check(
        self, levels: Literal["all"] | list[SCALES_TYPE]
    ) -> list[SCALES_TYPE]:
        r"""
        Check whether levels arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`),
        the user specifies which levels to include. This makes sure those
        levels are valid and gets them in the form we expect for the rest of
        the reconstruction. If the user passes `'all'`, this constructs the
        appropriate list (based on the values of `pyr_coeffs`).

        Parameters
        ----------
        levels :
            If `list` should contain some subset of integers from `0` to
            `self.num_scales-1` (inclusive) and `'residual_highpass'` and
            `'residual_lowpass'` (if appropriate for the pyramid). If `'all'`,
            returned value will contain all valid levels.

        Returns
        -------
        levels :
            List containing the valid levels for reconstruction.

        """
        if isinstance(levels, str):
            if levels != "all":
                raise TypeError(
                    "levels must be a list of levels or the string 'all' but"
                    f" got {levels}"
                )
            levels = (
                ["residual_highpass"]
                + list(range(self.num_scales))
                + ["residual_lowpass"]
            )
        else:
            if not hasattr(levels, "__iter__"):
                raise TypeError(
                    "levels must be a list of levels or the string 'all' but"
                    f" got {levels}"
                )
            levs_nums = np.array([int(i) for i in levels if isinstance(i, int)])
            assert (levs_nums >= 0).all(), "Level numbers must be non-negative."
            assert (levs_nums < self.num_scales).all(), (
                f"Level numbers must be in the range [0, {self.num_scales - 1:d}]"
            )
            levs_tmp = list(np.sort(levs_nums))  # we want smallest first
            if "residual_highpass" in levels:
                levs_tmp = ["residual_highpass"] + levs_tmp
            if "residual_lowpass" in levels:
                levs_tmp = levs_tmp + ["residual_lowpass"]
            levels = levs_tmp
        # not all pyramids have residual highpass / lowpass, but it's easier
        # to construct the list including them, then remove them if necessary.
        if "residual_lowpass" not in self.pyr_size and "residual_lowpass" in levels:
            levels.pop(-1)
        if "residual_highpass" not in self.pyr_size and "residual_highpass" in levels:
            levels.pop(0)
        return levels

    def _recon_bands_check(self, bands: Literal["all"] | list[int]) -> list[int]:
        """Check whether bands arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`),
        the user specifies which orientations to include. This makes sure those
        orientations are valid and gets them in the form we expect for the rest
        of the reconstruction. If the user passes `'all'`, this
        constructs the appropriate list (based on the values of `pyr_coeffs`).

        Parameters
        ----------
        bands :
            If list, should contain some subset of integers from `0` to
            `self.num_orientations-1`. If `'all'`, returned value will contain
            all valid orientations.

        Returns
        -------
        bands:
            List containing the valid orientations for reconstruction.

        """
        if isinstance(bands, str):
            if bands != "all":
                raise TypeError(
                    f"bands must be a list of ints or the string 'all' but got {bands}"
                )
            bands = np.arange(self.num_orientations)
        else:
            if not hasattr(bands, "__iter__"):
                raise TypeError(
                    f"bands must be a list of ints or the string 'all' but got {bands}"
                )
            bands: NDArray = np.array(bands, ndmin=1)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(), (
                "Error: band numbers must be in the range [0, "
                f"{self.num_orientations - 1:d}]"
            )
        return list(bands)

    def _recon_keys(
        self,
        levels: Literal["all"] | list[SCALES_TYPE],
        bands: Literal["all"] | list[int],
        max_orientations: int | None = None,
    ) -> list[KEYS_TYPE]:
        """Make a list of all the relevant keys from `pyr_coeffs` to use in pyramid
        reconstruction

        When reconstructing the input image (i.e., when calling `recon_pyr()`),
        the user specifies some subset of the pyramid coefficients to include
        in the reconstruction. This function takes in those specifications,
        checks that they're valid, and returns a list of tuples that are keys
        into the `pyr_coeffs` dictionary.

        Parameters
        ----------
        levels:
            If `list` should contain some subset of integers from `0` to
            `self.num_scales-1` (inclusive) and `'residual_highpass'` and
            `'residual_lowpass'` (if appropriate for the pyramid). If `'all'`,
            returned value will contain all valid levels.
        bands:
            If list, should contain some subset of integers from `0` to
            `self.num_orientations-1`. If `'all'`, returned value will contain
            all valid orientations.
        max_orientations:
            The maximum number of orientations we allow in the reconstruction.
            when we determine which ints are allowed for bands, we ignore all
            those greater than max_orientations.

        Returns
        -------
        recon_keys :
            List of `tuples`, all of which are keys in `pyr_coeffs`. These are
            the coefficients to include in the reconstruction of the image.

        """
        levels = self._recon_levels_check(levels)
        bands = self._recon_bands_check(bands)
        if max_orientations is not None:
            for i in bands:
                if i >= max_orientations:
                    warnings.warn(
                        f"You wanted band {i:d} in the reconstruction but"
                        f" max_orientation is {max_orientations:d}, so we"
                        "'re ignoring that band"
                    )
            bands = [i for i in bands if i < max_orientations]
        recon_keys = []
        for level in levels:
            # residual highpass and lowpass
            if isinstance(level, str):
                recon_keys.append(level)
            # else we have to get each of the (specified) bands at
            # that level
            else:
                recon_keys.extend([(level, band) for band in bands])
        return recon_keys

    def recon_pyr(
        self,
        pyr_coeffs: OrderedDict,
        levels: Literal["all"] | list[SCALES_TYPE] = "all",
        bands: Literal["all"] | list[int] = "all",
    ) -> Tensor:
        """Reconstruct the image or batch of images, optionally using subset of
        pyramid coefficients.

        NOTE: in order to call this function, you need to have
        previously called `self.forward(x)`, where `x` is the tensor you
        wish to reconstruct. This will fail if you called `forward()`
        with a subset of scales.

        Parameters
        ----------
        pyr_coeffs:
            pyramid coefficients to reconstruct from
        levels:
            If `list` should contain some subset of integers from `0` to
            `self.num_scales-1` (inclusive), `'residual_lowpass'`, and
            `'residual_highpass'`. If `'all'`, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.
        bands :
            If list, should contain some subset of integers from `0` to
            `self.num_orientations-1`. If `'all'`, returned value will contain
            all valid orientations. Otherwise, must be one of the valid
            orientations.

        Returns
        -------
        recon:
            The reconstructed image, of shape (batch, channel, height, width)

        """
        # For reconstruction to work, last time we called forward needed
        # to include all levels
        for s in self.scales:
            if isinstance(s, str):
                if s not in pyr_coeffs:
                    raise Exception(
                        f"scale {s} not in pyr_coeffs! pyr_coeffs must include"
                        " all scales, so make sure forward() was called with"
                        " arg scales=None"
                    )
            else:
                for b in range(self.num_orientations):
                    if (s, b) not in pyr_coeffs:
                        raise Exception(
                            f"scale {s} not in pyr_coeffs! pyr_coeffs must"
                            " include all scales, so make sure forward() was"
                            " called with arg scales=None"
                        )

        recon_keys = self._recon_keys(levels, bands)
        scale = 0

        # load masks from model
        lo0mask = self.lo0mask
        hi0mask = self.hi0mask

        # Recursively generate the reconstruction - function starts with
        # fine scales going down to coarse and then the reconstruction
        # is built recursively from the coarse scale up

        recondft = self._recon_levels(pyr_coeffs, recon_keys, scale)

        # generate highpass residual Reconstruction
        if "residual_highpass" in recon_keys:
            hidft = fft.fft2(
                pyr_coeffs["residual_highpass"],
                dim=(-2, -1),
                norm=self.fft_norm,
            )
            hidft = fft.fftshift(hidft)

            # output dft is the sum of the recondft from the recursive
            # function times the lomask (low pass component) with the
            # highpass dft * the highpass mask
            outdft = recondft * lo0mask + hidft * hi0mask
        else:
            outdft = recondft * lo0mask

        # get output reconstruction by inverting the fft
        reconstruction = fft.ifftshift(outdft)
        reconstruction = fft.ifft2(reconstruction, dim=(-2, -1), norm=self.fft_norm)

        # get real part of reconstruction (if complex)
        reconstruction = reconstruction.real

        return reconstruction

    def _recon_levels(
        self, pyr_coeffs: OrderedDict, recon_keys: list[KEYS_TYPE], scale: int
    ) -> Tensor:
        """Recursive function used to build the reconstruction. Called by recon_pyr

        Parameters
        ----------
        pyr_coeffs :
            Dictionary containing the coefficients of the pyramid. Keys are
            `(level, band)` tuples and the strings `'residual_lowpass'` and
            `'residual_highpass'` and values are Tensors of shape (batch,
            channel, height, width).
        recon_keys :
            list of the keys that index into the pyr_coeffs Dictionary
        scale :
            current scale that is being used to build the reconstruction
            scale is incremented by 1 on each call of the function

        Returns
        -------
        recondft :
            Current reconstruction based on the orientation band dft from the
            current scale summed with the output of recursive call with the
            next scale incremented

        """
        # base case, return the low-pass residual
        if scale == self.num_scales:
            if "residual_lowpass" in recon_keys:
                lodft = fft.fft2(
                    pyr_coeffs["residual_lowpass"],
                    dim=(-2, -1),
                    norm=self.fft_norm,
                )
                lodft = fft.fftshift(lodft)
            else:
                lodft = fft.fft2(
                    torch.zeros_like(pyr_coeffs["residual_lowpass"]),
                    dim=(-2, -1),
                    norm=self.fft_norm,
                )

            return lodft

        # Reconstruct from orientation bands
        # update himask
        himask = getattr(self, f"_himasks_scale_{scale}")
        orientdft = torch.zeros_like(pyr_coeffs[(scale, 0)])

        for b in range(self.num_orientations):
            if (scale, b) in recon_keys:
                anglemask = getattr(self, f"_anglemasks_recon_scale_{scale}")[b]
                coeffs = pyr_coeffs[(scale, b)]
                if self.tight_frame and self.is_complex:
                    coeffs = coeffs * np.sqrt(2)

                banddft = fft.fft2(coeffs, dim=(-2, -1), norm=self.fft_norm)
                banddft = fft.fftshift(banddft)

                complex_const = np.power(complex(0, 1), self.order)
                banddft = complex_const * banddft * anglemask * himask
                orientdft = orientdft + banddft

        # get the bounding box indices for the low-pass component
        lostart, loend = self._loindices[scale]

        # create lowpass mask
        lomask = getattr(self, f"_lomasks_scale_{scale}")

        # Recursively reconstruct by going to the next scale
        reslevdft = self._recon_levels(pyr_coeffs, recon_keys, scale + 1)
        # in not downsampled case, rescale the magnitudes of the reconstructed
        # dft at each level by factor of 2 to account for the scaling in the forward
        if (not self.tight_frame) and (not self.downsample):
            reslevdft = reslevdft / 2
        # create output for reconstruction result
        resdft = torch.zeros_like(pyr_coeffs[(scale, 0)], dtype=torch.complex64)

        # place upsample and convolve lowpass component
        resdft[:, :, lostart[0] : loend[0], lostart[1] : loend[1]] = reslevdft * lomask
        recondft = resdft + orientdft
        # add orientation interpolated and added images to the lowpass image
        return recondft

    def steer_coeffs(
        self,
        pyr_coeffs: OrderedDict,
        angles: list[float],
        even_phase: bool = True,
    ) -> tuple[dict, dict]:
        """Steer pyramid coefficients to the specified angles

        This allows you to have filters that have the Gaussian derivative order
        specified in construction, but arbitrary angles or number of orientations.

        Parameters
        ----------
        pyr_coeffs :
            the pyramid coefficients to steer
        angles :
            list of angles (in radians) to steer the pyramid coefficients to
        even_phase :
            specifies whether the harmonics are cosine or sine phase aligned
            about those positions.

        Returns
        -------
        resteered_coeffs :
            dictionary of re-steered pyramid coefficients. will have the same
            number of scales as the original pyramid (though it will not
            contain the residual highpass or lowpass). like `pyr_coeffs`, keys
            are 2-tuples of ints indexing the scale and orientation, but now
            we're indexing `angles` instead of `self.num_orientations`.
        resteering_weights :
            dictionary of weights used to re-steer the pyramid coefficients.
            will have the same keys as `resteered_coeffs`.

        """
        assert pyr_coeffs[(0, 0)].dtype not in complex_types, (
            "steering only implemented for real coefficients"
        )
        resteered_coeffs = {}
        resteering_weights = {}
        num_scales = self.num_scales
        num_orientations = self.num_orientations
        for i in range(num_scales):
            basis = torch.cat(
                [
                    pyr_coeffs[(i, j)].squeeze().unsqueeze(-1)
                    for j in range(num_orientations)
                ],
                dim=-1,
            )

            for j, a in enumerate(angles):
                res, steervect = steer(
                    basis, a, return_weights=True, even_phase=even_phase
                )
                resteering_weights[(i, j)] = steervect
                resteered_coeffs[(i, num_orientations + j)] = res.reshape(
                    pyr_coeffs[(i, 0)].shape
                )

        return resteered_coeffs, resteering_weights
