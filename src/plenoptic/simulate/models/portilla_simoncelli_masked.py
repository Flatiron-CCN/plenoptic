"""Portilla-Simoncelli texture statistics.

The Portilla-Simoncelli (PS) texture statistics are a set of image
statistics, first described in [1]_, that are proposed as a sufficient set
of measurements for describing visual textures. That is, if two texture
images have the same values for all PS texture stats, humans should
consider them as members of the same family of textures.
"""

from collections import OrderedDict
from typing import Literal

import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft
import torch.nn as nn
from torch import Tensor

from ...tools import signal
from ...tools.conv import blur_downsample
from ...tools.data import to_numpy
from ...tools.display import clean_stem_plot, clean_up_axes, update_stem
from ...tools.validate import validate_input
from ..canonical_computations.steerable_pyramid_freq import (
    SCALES_TYPE as PYR_SCALES_TYPE,
)
from ..canonical_computations.steerable_pyramid_freq import SteerablePyramidFreq

SCALES_TYPE = Literal["pixel_statistics"] | PYR_SCALES_TYPE


class PortillaSimoncelliMasked(nn.Module):
    r"""Portila-Simoncelli texture statistics.

    The Portilla-Simoncelli (PS) texture statistics are a set of image
    statistics, first described in [1]_, that are proposed as a sufficient set
    of measurements for describing visual textures. That is, if two texture
    images have the same values for all PS texture stats, humans should
    consider them as members of the same family of textures.

    The PS stats are computed based on the steerable pyramid [2]_. They consist
    of the local auto-correlations, cross-scale (within-orientation)
    correlations, and cross-orientation (within-scale) correlations of both the
    pyramid coefficients and the local energy (as computed by those
    coefficients). Additionally, they include the first four global moments
    (mean, variance, skew, and kurtosis) of the image and down-sampled versions
    of that image. See the paper and notebook for more description.

    Parameters
    ----------
    image_shape:
        Shape of input image.
    mask:
        List of 3d tensors. We use the masks to perform sums, and so the masks should be
        normalized in order to perform the averages. See tutorial for example.
    n_scales:
        The number of pyramid scales used to measure the statistics (default=4)
    n_orientations:
        The number of orientations used to measure the statistics (default=4)
    spatial_corr_width:
        The width of the spatial cross- and auto-correlation statistics

    Attributes
    ----------
    scales: list
        The names of the unique scales of coefficients in the pyramid, used for
        coarse-to-fine metamer synthesis.

    References
    ----------
    .. [1] J Portilla and E P Simoncelli. A Parametric Texture Model based on
       Joint Statistics of Complex Wavelet Coefficients. Int'l Journal of
       Computer Vision. 40(1):49-71, October, 2000.
       https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla99-abstract.html
       https://www.cns.nyu.edu/~lcv/texture/
    .. [2] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible
       Architecture for Multi-Scale Derivative Computation," Second Int'l Conf
       on Image Processing, Washington, DC, Oct 1995.

    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        mask: list[Tensor],
        n_scales: int = 4,
        n_orientations: int = 4,
        spatial_corr_width: int = 9,
    ):
        super().__init__()

        self.image_shape = image_shape
        if any([(image_shape[-1] / 2**i) % 2 for i in range(n_scales)]) or any(
            [(image_shape[-2] / 2**i) % 2 for i in range(n_scales)]
        ):
            raise ValueError(
                "Because of how the Portilla-Simoncelli model handles "
                "multiscale representations, it only works with images"
                " whose shape can be divided by 2 `n_scales` times."
            )
        if any([m.ndim != 3 for m in mask]):
            raise ValueError("All masks must be 3d!")
        if any([m.shape[-2:] != image_shape for m in mask]):
            raise ValueError(
                "Last two dimensions of mask must be height and width"
                " and must match image_shape!"
            )
        if any([m.min() < 0 for m in mask]):
            raise ValueError("All masks must be non-negative!")
        # we need to downsample the masks for each scale, plus one additional scale for
        # the reconstructed lowpass image
        for i in range(n_scales + 1):
            if i == 0:
                scale_mask = mask
            else:
                # multiply by the factor of four in order to keep the sum
                # approximately equal across scales.
                scale_mask = [
                    4 ** (i / len(mask))
                    * blur_downsample(m.unsqueeze(0), i, scale_filter=True).squeeze(0)
                    for m in mask
                ]
            for j, m in enumerate(scale_mask):
                # it's possible negative values will get introduced by the downsampling
                # above, in which case we remove them, since they mess up our
                # computations. in particular, they could result in negative variance
                # values.
                self.register_buffer(f"_mask_{j}_scale_{i}", m.clip(min=0))
        # these indices are used to create the einsum expressions
        self._mask_input_idx = ", ".join([f"m{i} h w" for i in range(len(mask))])
        self._n_masks = len(mask)
        self._mask_output_idx = f"{' '.join([f'm{i}' for i in range(len(mask))])}"
        self.spatial_corr_width = spatial_corr_width
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        # these are each lists of tensors of shape (batch, channel, n_autocorrs, height,
        # width), one per scale, where n_autocorrs is approximately
        # spatial_corr_width^2 / 2
        rolls_h, rolls_w = self._create_autocorr_idx(spatial_corr_width, image_shape)
        for i, (h, w) in enumerate(zip(rolls_h, rolls_w)):
            self.register_buffer(f"_autocorr_rolls_h_scale_{i}", h)
            self.register_buffer(f"_autocorr_rolls_w_scale_{i}", w)
        self._n_autocorrs = rolls_h[0].shape[3]
        self._pyr = SteerablePyramidFreq(
            self.image_shape,
            height=self.n_scales,
            order=self.n_orientations - 1,
            is_complex=True,
            tight_frame=False,
        )

        self.scales = (
            ["pixel_statistics", "residual_lowpass"]
            + [ii for ii in range(n_scales - 1, -1, -1)]
            + ["residual_highpass"]
        )

        # Dictionary defining shape of the statistics and which scale they're
        # associated with
        scales_shape_dict = self._create_scales_shape_dict()

        # Dictionary defining necessary statistics, that is, those that are not
        # redundant
        self._necessary_stats_dict = self._create_necessary_stats_dict(
            scales_shape_dict
        )
        # turn this into tensor we can use in forward pass. first into a
        # boolean mask...
        _necessary_stats_mask = einops.pack(
            list(self._necessary_stats_dict.values()), "*"
        )[0]
        # then into a tensor of indices
        _necessary_stats_mask = torch.where(_necessary_stats_mask)[0]
        self.register_buffer("_necessary_stats_mask", _necessary_stats_mask)

        # This array is composed of the following values: 'pixel_statistics',
        # 'residual_lowpass', 'residual_highpass' and integer values from 0 to
        # self.n_scales-1. It is the same size as the representation tensor
        # returned by this object's forward method. It must be a numpy array so
        # we can have a mixture of ints and strs (and so we can use np.in1d
        # later)
        self._representation_scales = einops.pack(
            list(scales_shape_dict.values()), "*"
        )[0]
        # just select the scales of the necessary stats.
        self._representation_scales = self._representation_scales[
            self._necessary_stats_mask
        ]
        # There are two types of computations where we add a little epsilon to help with
        # stability:
        # - division of one statistic by another
        # - taking the sqrt of one statistic
        # In both cases, adding a small epsilon avoids a NaN or (for division) an
        # unreasonably large number.
        self._stability_epsilon = 1e-6
        # this (much larger) epsilon is used when computing the pixel stats skew and
        # kurtosis, which involve a division by the variance to the 1.5 and 2 powers,
        # respectively, and have much greater problems with very small values.
        self._pixel_epsilon = 1e-1

    @property
    def mask(self):
        # inspired by
        # https://discuss.pytorch.org/t/why-no-nn-bufferlist-like-function-for-registered-buffer-tensor/18884/10
        return [
            [getattr(self, f"_mask_{j}_scale_{i}") for j in range(self._n_masks)]
            for i in range(self.n_scales + 1)
        ]

    @property
    def _autocorr_rolls_h(self):
        # inspired by
        # https://discuss.pytorch.org/t/why-no-nn-bufferlist-like-function-for-registered-buffer-tensor/18884/10
        return [
            getattr(self, f"_autocorr_rolls_h_scale_{i}")
            for i in range(self.n_scales + 1)
        ]

    @property
    def _autocorr_rolls_w(self):
        # inspired by
        # https://discuss.pytorch.org/t/why-no-nn-bufferlist-like-function-for-registered-buffer-tensor/18884/10
        return [
            getattr(self, f"_autocorr_rolls_w_scale_{i}")
            for i in range(self.n_scales + 1)
        ]

    def _create_autocorr_idx(
        self, spatial_corr_width, image_shape
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Create indices used to shift images when computing autocorrelation.

        The autocorrelation of ``img`` is the product of ``img`` with itself shifted by
        a small number of pixels. That is: ``einops.einsum(img, img.roll(i, -1).roll(j,
        -2))`` for some relevant values of i and j. This method computes the indices
        corresponding to those rolls, so that we can simply call ``img.gather(rolls_h,
        -2).gather(rolls_w, -1)`` during the forward pass instead of ``img.roll(i,
        -1).roll(j, -2)``, which is less efficient.

        Because of the symmetry of autocorrelations (see Portilla-Simoncelli notebook
        for details), we do not need the full ``spatial_corr_width**2`` shifts, we only
        need everything below the diagonal (e.g., we don't need to roll both 1 pixel to
        the left and 1 pixel to the right).

        Parameters
        ----------
        spatial_corr_width :
            The width of the spatial auto-correlation.
        image_shape :
            Shape of input image.

        Returns
        -------
        rolls_h, rolls_w :
            List of tensors of shape ``(1, 1, n_orientations, n_autocorrs, height,
            width)`` giving the shifts along the height (``shape[-2]``) and width
            (``shape[-1]``) dimensions required for computing the autocorrelations. Each
            entry in the list corresponds to a different scale, and thus height and
            width decrease.

        """
        # because of the symmetry of autocorrelation, in order to generate all
        # autocorrelations, we only need the lower triangle (so that we take the
        # autocorrelation between the image and itself shifted 1 pixel to the left, but
        # not also shifted 1 pixel to the right)...
        half_width = (spatial_corr_width - 1) // 2
        autocorr_shift_vals = [
            i - half_width for i in np.tril_indices(spatial_corr_width)
        ]
        # if spatial_corr_width is even, then we also need these shifts:
        if np.mod(spatial_corr_width, 2) == 0:
            autocorr_shift_vals[0] = np.concatenate(
                [
                    np.zeros(spatial_corr_width, dtype=int) - half_width,
                    autocorr_shift_vals[0],
                ],
                0,
            )
            autocorr_shift_vals[1] = np.concatenate(
                [
                    np.arange(spatial_corr_width, dtype=int) - half_width,
                    autocorr_shift_vals[1],
                ],
                0,
            )
        # and up to the central element on the diagonal.
        idx = [i != j or i < 0 for i, j in zip(*autocorr_shift_vals)]
        # put the (0, 0) shift, which corresponds to the variance, at the very end, so
        # we know where it is
        autocorr_shift_vals = [
            np.concatenate([i[idx], np.zeros(1, dtype=int)], 0)
            for i in autocorr_shift_vals
        ]

        img_h, img_w = image_shape
        rolls_h, rolls_w = [], []
        # need one additional scale, since we compute the autocorrelation of the
        # reconstructed lowpass images as well
        for _ in range(self.n_scales + 1):
            arange_h = (
                torch.arange(img_h)
                .view((1, 1, 1, img_h, 1))
                .repeat((1, 1, self.n_orientations, 1, img_h))
            )
            arange_w = (
                torch.arange(img_w)
                .view((1, 1, 1, 1, img_w))
                .repeat((1, 1, self.n_orientations, img_w, 1))
            )
            rolls_h.append(
                torch.stack([arange_h.roll(i, -2) for i in autocorr_shift_vals[0]], 3)
            )
            rolls_w.append(
                torch.stack([arange_w.roll(i, -1) for i in autocorr_shift_vals[1]], 3)
            )
            img_h = int(img_h // 2)
            img_w = int(img_w // 2)
        return rolls_h, rolls_w

    def _create_scales_shape_dict(self) -> OrderedDict:
        """Create dictionary defining scales and shape of each stat.

        This dictionary functions as metadata which is used for two main
        purposes:

        - Scale assignment. In order for optimization to work well, we proceed
          in a "coarse-to-fine" manner. That is, we start optimization by only
          considering the statistics related to the lowest frequencies, and
          gradually add in those related to higher and higher frequencies. This
          is similar to blurring the objective function and then gradually
          adding in finer and finer details. The numbers in this dictionary map
          the computed statistics to their corresponding scales, which we use
          in remove_scales to throw away some stats as needed.

        - Redundant stat identification. As described at the bottom of the
          notebook, the model incidentally computes a whole bunch of redundant
          stats, because auto- and cross-correlation matrices have certain
          symmetries. the _create_necessary_stats_dict method accepts the
          dictionary created here as input and uses the values to get the
          shapes of these and insert True/False as necessary.

        Returns
        -------
        scales_shape_dict
           Dictionary defining shape and associated scales of each computed
           statistic. The keys name each statistic, with dummy arrays as
           values. These arrays have the same shape as the stat (excluding
           batch and channel), with values defining which scale they correspond
           to.

        """
        shape_dict = OrderedDict()
        # There are 6 pixel statistics
        shape_dict["pixel_statistics"] = np.array(4 * ["pixel_statistics"])

        # These are the basic building blocks of the scale assignments for many
        # of the statistics calculated by the PortillaSimoncelli model.
        scales = np.arange(self.n_scales)
        # the cross-scale correlations exclude the coarsest scale
        scales_without_coarsest = np.arange(self.n_scales - 1)
        # the statistics computed on the reconstructed bandpass images have an
        # extra scale corresponding to the lowpass residual
        scales_with_lowpass = np.array(
            scales.tolist() + ["residual_lowpass"], dtype=object
        )

        # now we go through each statistic in order and create a dummy array
        # full of 1s with the same shape as the actual statistic (excluding the
        # batch and channel dimensions, as each stat is computed independently
        # across those dimensions). We then multiply it by one of the scales
        # arrays above to turn those 1s into values describing the
        # corresponding scale.

        auto_corr_mag = np.ones(
            (self._n_autocorrs - 1, self.n_orientations, self.n_scales), dtype=int
        )
        # this rearrange call is turning scales from 1d with shape (n_scales, )
        # to 4d with shape (1, 1, n_scales, 1), so that it matches
        # auto_corr_mag. the following rearrange calls do similar.
        auto_corr_mag *= einops.rearrange(scales, "s -> 1 1 s")
        shape_dict["auto_correlation_magnitude"] = auto_corr_mag

        shape_dict["skew_reconstructed"] = scales_with_lowpass

        shape_dict["kurtosis_reconstructed"] = scales_with_lowpass

        auto_corr = np.ones((self._n_autocorrs - 1, self.n_scales + 1), dtype=object)
        auto_corr *= einops.rearrange(scales_with_lowpass, "s -> 1 s")
        shape_dict["auto_correlation_reconstructed"] = auto_corr

        shape_dict["std_reconstructed"] = scales_with_lowpass

        cross_orientation_corr_mag = np.ones(
            (self.n_orientations, self.n_orientations, self.n_scales), dtype=int
        )
        cross_orientation_corr_mag *= einops.rearrange(scales, "s -> 1 1 s")
        shape_dict["cross_orientation_correlation_magnitude"] = (
            cross_orientation_corr_mag
        )

        mags_std = np.ones((self.n_orientations, self.n_scales), dtype=int)
        mags_std *= einops.rearrange(scales, "s -> 1 s")
        shape_dict["magnitude_std"] = mags_std

        cross_scale_corr_mag = np.ones(
            (self.n_orientations, self.n_orientations, self.n_scales - 1), dtype=int
        )
        cross_scale_corr_mag *= einops.rearrange(scales_without_coarsest, "s -> 1 1 s")
        shape_dict["cross_scale_correlation_magnitude"] = cross_scale_corr_mag

        cross_scale_corr_real = np.ones(
            (self.n_orientations, 2 * self.n_orientations, self.n_scales - 1), dtype=int
        )
        cross_scale_corr_real *= einops.rearrange(scales_without_coarsest, "s -> 1 1 s")
        shape_dict["cross_scale_correlation_real"] = cross_scale_corr_real

        shape_dict["var_highpass_residual"] = np.array(["residual_highpass"])

        return shape_dict

    def _create_necessary_stats_dict(
        self, scales_shape_dict: OrderedDict
    ) -> OrderedDict:
        """Create mask specifying the necessary statistics.

        Some of the statistics computed by the model are redundant, due to
        symmetries. For example, about half of the values in the
        autocorrelation matrices are duplicates. See the Portilla-Simoncelli
        notebook for more details.

        Parameters
        ----------
        scales_shape_dict
            Dictionary defining shape and associated scales of each computed
            statistic.

        Returns
        -------
        necessary_stats_dict
            Dictionary defining which statistics are necessary (i.e., not
            redundant). Will have the same keys as scales_shape_dict, with the
            values being boolean tensors of the same shape as
            scales_shape_dict's corresponding values. True denotes the
            statistics that will be included in the model's output, while False
            denotes the redundant ones we will toss.

        """
        mask_dict = scales_shape_dict.copy()
        # Upper triangle indices, including diagonal. These are redundant stats
        # for cross_orientation_correlation_magnitude (because we've normalized
        # this matrix to be true cross-correlations, the diagonals are all 1,
        # like for the auto-correlations)
        triu_inds = torch.triu_indices(self.n_orientations, self.n_orientations)
        for k, v in mask_dict.items():
            if k == "cross_orientation_correlation_magnitude":
                # Symmetry M_{i,j} = M_{j,i}.
                # Start with all True, then place False in redundant stats.
                mask = torch.ones(v.shape, dtype=torch.bool)
                mask[triu_inds[0], triu_inds[1]] = False
            else:
                # all of the other stats have no redundancies
                mask = torch.ones(v.shape, dtype=torch.bool)
            mask_dict[k] = mask
        return mask_dict

    def forward(self, image: Tensor, scales: list[SCALES_TYPE] | None = None) -> Tensor:
        r"""Generate Texture Statistics representation of an image.

        Note that separate batches and channels are analyzed in parallel.

        Parameters
        ----------
        image :
            A 4d tensor (batch, channel, height, width) containing the image(s) to
            analyze.
        scales :
            Which scales to include in the returned representation. If None, we
            include all scales. Otherwise, can contain subset of values present
            in this model's ``scales`` attribute, and the returned tensor will
            then contain the subset corresponding to those scales.

        Returns
        -------
        representation_tensor:
            3d tensor of shape (batch, channel, stats) containing the measured
            texture statistics.

        Raises
        ------
        ValueError :
            If `image` is not 4d or has a dtype other than float or complex.

        """
        validate_input(image)

        # pyr_dict is the dictionary of complex-valued tensors returned by the
        # steerable pyramid. pyr_coeffs is a list (length n_scales) of 5d
        # tensors, each of shape (batch, channel, scales, n_orientations,
        # height, width) containing the complex-valued oriented bands, while
        # highpass is a real-valued 4d tensor of shape (batch, channel, height,
        # width). Note that the residual lowpass in pyr_dict has been demeaned.
        # We keep both the dict and list of pyramid coefficients because we
        # need the dictionary for reconstructing the image done later on.
        pyr_dict, pyr_coeffs, highpass, _ = self._compute_pyr_coeffs(image)

        # Now, we create several intermediate representations that we'll use to
        # compute the texture statistics later.

        # First, two intermediate dictionaries: magnitude_pyr_coeffs and
        # real_pyr_coeffs, which contain the demeaned magnitude of the pyramid
        # coefficients and the real part of the pyramid coefficients
        # respectively.
        mag_pyr_coeffs, real_pyr_coeffs = self._compute_intermediate_representations(
            pyr_coeffs
        )

        # Then, the reconstructed lowpass image at each scale. (this is a list
        # of length n_scales+1 containing tensors of shape (batch, channel,
        # height, width))
        reconstructed_images = self._reconstruct_lowpass_at_each_scale(pyr_dict)
        # the reconstructed_images list goes from coarse-to-fine, but we want
        # each of the stats computed from it to go from fine-to-coarse, so we
        # reverse its direction.
        reconstructed_images = reconstructed_images[::-1]

        # Now, start calculating the PS texture stats.

        # Calculate pixel statistics (mean, variance, skew, kurtosis). This is a tensor
        # of shape (batch, channel, masks, 4)
        pixel_stats = self._compute_pixel_stats(self.mask[0], image)

        # Compute the central autocorrelation of the coefficient magnitudes. This is a
        # tensor of shape: (batch, channel, masks, n_autocorrs, n_orientations,
        # n_scales).
        autocorr_mags, mags_var = self._compute_autocorr(self.mask, mag_pyr_coeffs)
        # mags_var is the variance of the magnitude coefficients at each scale (it's an
        # intermediary of the computation of the auto-correlations). We take the square
        # root to get the standard deviation. After this, mags_std will have shape
        # (batch, channel, masks, n_orientations, n_scale)
        mags_std = einops.rearrange(
            (mags_var + self._stability_epsilon).sqrt(),
            f"b c {self._mask_output_idx} o s -> b c ({self._mask_output_idx}) o s",
        )

        # Compute the central autocorrelation of the reconstructed lowpass images at
        # each scale (and their variances). autocorr_recon is a tensor of shape (batch,
        # channel, masks, n_autocorrs, n_scales+1), and var_recon is a tensor of shape
        # (batch, channel, masks, n_scales+1)
        autocorr_recon, var_recon = self._compute_autocorr(
            self.mask, reconstructed_images
        )
        # Compute the standard deviation, skew, and kurtosis of each reconstructed
        # lowpass image. std_recon, skew_recon, and kurtosis_recon will all end up as
        # tensors of shape (batch, channel, masks, n_scales+1)
        std_recon = einops.rearrange(
            (var_recon + self._stability_epsilon).sqrt(),
            f"b c {self._mask_output_idx} s -> b c ({self._mask_output_idx}) s",
        )
        skew_recon, kurtosis_recon = self._compute_skew_kurtosis_recon(
            self.mask, reconstructed_images, var_recon
        )

        # Compute the cross-orientation correlations between the magnitude
        # coefficients at each scale. this will be a tensor of shape (batch,
        # channel, n_orientations, n_orientations, n_scales)
        cross_ori_corr_mags = self._compute_cross_correlation(
            self.mask, mag_pyr_coeffs, mag_pyr_coeffs, mags_var, mags_var
        )

        # If we have more than one scale, compute the cross-scale correlations
        if self.n_scales != 1:
            # First, double the phase the coefficients, so we can correctly
            # compute correlations across scales.
            phase_doubled_mags, phase_doubled_sep = self._double_phase_pyr_coeffs(
                pyr_coeffs
            )
            # Compute the cross-scale correlations between the magnitude
            # coefficients. For each coefficient, we're correlating it with the
            # coefficients at the next-coarsest scale. this will be a tensor of
            # shape (batch, channel, n_orientations, n_orientations,
            # n_scales-1)
            cross_scale_corr_mags = self._compute_cross_correlation(
                self.mask, mag_pyr_coeffs[:-1], phase_doubled_mags, mags_var[..., :-1]
            )
            # Compute the cross-scale correlations between the real
            # coefficients and the real and imaginary coefficients at the next
            # coarsest scale. this will be a tensor of shape (batch, channel,
            # n_orientations, 2*n_orientations, n_scales-1)
            cross_scale_corr_real = self._compute_cross_correlation(
                self.mask, real_pyr_coeffs[:-1], phase_doubled_sep
            )

        # Compute the variance of the highpass residual
        var_highpass_residual = einops.einsum(
            *self.mask[0],
            highpass.pow(2),
            f"{self._mask_input_idx}, b c h w -> b c {self._mask_output_idx}",
        )
        var_highpass_residual = einops.rearrange(
            var_highpass_residual,
            f"b c {self._mask_output_idx} -> b c ({self._mask_output_idx})",
        )

        # Now, combine all these stats together, first into a list
        all_stats = [
            pixel_stats,
            autocorr_mags,
            skew_recon,
            kurtosis_recon,
            autocorr_recon,
            std_recon,
            cross_ori_corr_mags,
            mags_std,
        ]
        if self.n_scales != 1:
            all_stats += [cross_scale_corr_mags, cross_scale_corr_real]
        all_stats += [var_highpass_residual]
        # And then pack them into a 3d tensor
        representation_tensor, pack_info = einops.pack(all_stats, "b c m *")

        # the only time when this is None is during testing, when we make sure
        # that our assumptions are all valid.
        if self._necessary_stats_mask is None:
            # store this so we can unpack this info (only possible when we've
            # discarded no stats)
            self._pack_info = pack_info
        else:
            # Throw away all redundant statistics
            representation_tensor = representation_tensor.index_select(
                -1, self._necessary_stats_mask
            )

        # Return the subset of stats corresponding to the specified scale.
        if scales is not None:
            representation_tensor = self.remove_scales(representation_tensor, scales)

        return representation_tensor

    def remove_scales(
        self, representation_tensor: Tensor, scales_to_keep: list[SCALES_TYPE]
    ) -> Tensor:
        """Remove statistics not associated with scales.

        For a given representation_tensor and a list of scales_to_keep, this
        attribute removes all statistics *not* associated with those scales.

        Note that calling this method will always remove statistics.

        Parameters
        ----------
        representation_tensor:
            3d tensor containing the measured representation statistics.
        scales_to_keep:
            Which scales to include in the returned representation. Can contain
            subset of values present in this model's ``scales`` attribute, and
            the returned tensor will then contain the subset of the full
            representation corresponding to those scales.

        Returns
        -------
        limited_representation_tensor :
            Representation tensor with some statistics removed.

        """
        # this is necessary because object is the dtype of
        # self._representation_scales
        scales_to_keep = np.array(scales_to_keep, dtype=object)
        # np.in1d returns a 1d boolean array of the same shape as
        # self._representation_scales with True at each location where that
        # value appears in scales_to_keep. where then converts this boolean
        # array into indices
        ind = np.where(np.in1d(self._representation_scales, scales_to_keep))[0]
        ind = torch.from_numpy(ind).to(representation_tensor.device)
        return representation_tensor.index_select(-1, ind)

    def convert_to_tensor(self, representation_dict: OrderedDict) -> Tensor:
        r"""Convert dictionary of statistics to a tensor.

        Parameters
        ----------
        representation_dict :
             Dictionary of representation.

        Returns
        -------
        3d tensor of statistics.

        See Also
        --------
        convert_to_dict:
            Convert tensor representation to dictionary.

        """
        rep = einops.pack(list(representation_dict.values()), "b c *")[0]
        # then get rid of all the nans / unnecessary stats
        return rep.index_select(-1, self._necessary_stats_mask)

    def convert_to_dict(self, representation_tensor: Tensor) -> OrderedDict:
        """Convert tensor of statistics to a dictionary.

        While the tensor representation is required by plenoptic's synthesis
        objects, the dictionary representation is easier to manually inspect.

        This dictionary will contain NaNs in its values: these are placeholders
        for the redundant statistics.

        Parameters
        ----------
        representation_tensor
            3d tensor of statistics.

        Returns
        -------
        rep
            Dictionary of representation, with informative keys.

        See Also
        --------
        convert_to_tensor:
            Convert dictionary representation to tensor.

        """
        if representation_tensor.shape[-1] != len(self._representation_scales):
            raise ValueError(
                "representation tensor is the wrong length (expected "
                f"{len(self._representation_scales)} but got"
                f"{representation_tensor.shape[-1]})!"
                " Did you remove some of the scales? (i.e., by setting "
                "scales in the forward pass)? convert_to_dict does not "
                "support such tensors."
            )

        rep = self._necessary_stats_dict.copy()
        n_filled = 0
        for k, v in rep.items():
            # each statistic is a tensor with batch and channel dimensions as
            # found in representation_tensor and all the other dimensions
            # determined by the values in necessary_stats_dict.
            shape = (*representation_tensor.shape[:3], *v.shape)
            new_v = torch.nan * torch.ones(
                shape,
                dtype=representation_tensor.dtype,
                device=representation_tensor.device,
            )
            # v.sum() gives the number of necessary elements from this stat
            this_stat_vec = representation_tensor[..., n_filled : n_filled + v.sum()]
            # use boolean indexing to put the values from new_stat_vec in the
            # appropriate place
            new_v[..., v] = this_stat_vec
            rep[k] = new_v
            n_filled += v.sum()
        return rep

    def _compute_pyr_coeffs(
        self, image: Tensor
    ) -> tuple[OrderedDict, list[Tensor], Tensor, Tensor]:
        """Compute pyramid coefficients of image.

        Note that the residual lowpass has been demeaned independently for each
        batch and channel (and this is true of the lowpass returned separately
        as well as the one included in pyr_coeffs_dict)

        Parameters
        ----------
        image :
            4d tensor of shape (batch, channel, height, width) containing the
            image

        Returns
        -------
        pyr_coeffs_dict :
            OrderedDict of containing all pyramid coefficients.
        pyr_coeffs :
            List of length n_scales, containing 5d tensors of shape (batch,
            channel, n_orientations, height, width) containing the complex-valued
            oriented bands (note that height and width shrink by half on each
            scale). This excludes the residual highpass and lowpass bands.
        highpass :
            The residual highpass as a real-valued 4d tensor (batch, channel,
            height, width)
        lowpass :
            The residual lowpass as a real-valued 4d tensor (batch, channel,
            height, width). This tensor has been demeaned (independently for
            each batch and channel).

        """
        pyr_coeffs = self._pyr.forward(image)
        # separate out the residuals and demean the residual lowpass
        lowpass = pyr_coeffs["residual_lowpass"]
        lowpass = lowpass - lowpass.mean(dim=(-2, -1), keepdim=True)
        pyr_coeffs["residual_lowpass"] = lowpass
        highpass = pyr_coeffs["residual_highpass"]

        # This is a list of tensors, one for each scale, where each tensor is
        # of shape (batch, channel, n_orientations, height, width) (note that
        # height and width halves on each scale)
        coeffs_list = [
            torch.stack([pyr_coeffs[(i, j)] for j in range(self.n_orientations)], 2)
            for i in range(self.n_scales)
        ]
        return pyr_coeffs, coeffs_list, highpass, lowpass

    def _compute_pixel_stats(self, mask: list[Tensor], image: Tensor) -> Tensor:
        """Compute the pixel stats: first four moments.

        Note that for the masked version, these are the *non-central* moments, i.e.,
        they're just the image raised to the first through fourth powers.

        Parameters
        ----------
        mask :
            The mask to use for weighting.
        image :
            4d tensor of shape (batch, channel, height, width) containing input
            image. Stats are computed indepently for each batch and channel.

        Returns
        -------
        pixel_stats :
            4d tensor of shape (batch, channel, masks, 4) containing the first four
            non-central moments

        """
        weighted_avg_expr = (
            f"{self._mask_input_idx}, b c h w -> b c {self._mask_output_idx}"
        )
        mean = einops.einsum(*mask, image, weighted_avg_expr)
        # these are all non-central moments...
        moment_2 = einops.einsum(*mask, image.pow(2), weighted_avg_expr)
        moment_3 = einops.einsum(*mask, image.pow(3), weighted_avg_expr)
        moment_4 = einops.einsum(*mask, image.pow(4), weighted_avg_expr)
        # ... which we use to compute the var, skew, and kurtosis. the formulas we use
        # for var and skew here can be found on their respective wikipedia pages, and
        # the one for kurtosis comes from Eero working through the algebra
        var = moment_2 - mean.pow(2)
        skew = (moment_3 - 3 * mean * var - mean.pow(3)) / (
            var.pow(1.5) + self._pixel_epsilon
        )
        kurtosis = (
            moment_4
            - 4 * mean * moment_3
            + 6 * mean.pow(2) * moment_2
            - 3 * mean.pow(4)
        ) / (var.pow(2) + self._pixel_epsilon)
        return einops.rearrange(
            [mean, var, skew, kurtosis],
            f"stats b c {self._mask_output_idx} -> b c ({self._mask_output_idx}) stats",
        )

    @staticmethod
    def _compute_intermediate_representations(
        pyr_coeffs: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Compute useful intermediate representations.

        These representations are:
          1) demeaned magnitude of the pyramid coefficients,
          2) real part of the pyramid coefficients

        These two are used in computing some of the texture representation.

        Parameters
        ----------
        pyr_coeffs :
            Complex steerable pyramid coefficients (without residuals), as list
            of length n_scales, containing 5d tensors of shape (batch, channel,
            n_orientations, height, width)

        Returns
        -------
        magnitude_pyr_coeffs :
           List of length n_scales, containing 5d tensors of shape (batch,
           channel, n_orientations, height, width) (same as ``pyr_coeffs``),
           containing the demeaned magnitude of the steerable pyramid
           coefficients (i.e., coeffs.abs() - coeffs.abs().mean((-2, -1)))
        real_pyr_coeffs :
           List of length n_scales, containing 5d tensors of shape (batch,
           channel, n_orientations, height, width) (same as ``pyr_coeffs``),
           containing the real components of the coefficients (i.e.
           coeffs.real)

        """
        magnitude_pyr_coeffs = [coeff.abs() for coeff in pyr_coeffs]
        magnitude_means = [
            mag.mean((-2, -1), keepdim=True) for mag in magnitude_pyr_coeffs
        ]
        magnitude_pyr_coeffs = [
            mag - mn for mag, mn in zip(magnitude_pyr_coeffs, magnitude_means)
        ]
        real_pyr_coeffs = [coeff.real for coeff in pyr_coeffs]
        return magnitude_pyr_coeffs, real_pyr_coeffs

    def _reconstruct_lowpass_at_each_scale(
        self, pyr_coeffs_dict: OrderedDict
    ) -> list[Tensor]:
        """Reconstruct the lowpass unoriented image at each scale.

        The autocorrelation, standard deviation, skew, and kurtosis of each of
        these images is part of the texture representation.

        Parameters
        ----------
        pyr_coeffs_dict :
            Dictionary containing the steerable pyramid coefficients, with the
            lowpass residual demeaned.

        Returns
        -------
        reconstructed_images :
            List of length n_scales+1 containing the reconstructed unoriented
            image at each scale, from fine to coarse. The final image is
            reconstructed just from the residual lowpass image. Each is a 4d
            tensor, this is a list because they are all different heights and
            widths.

        """
        reconstructed_images = [
            self._pyr.recon_pyr(pyr_coeffs_dict, levels=["residual_lowpass"])
        ]
        # go through scales backwards
        for lev in range(self.n_scales - 1, -1, -1):
            recon = self._pyr.recon_pyr(pyr_coeffs_dict, levels=[lev])
            reconstructed_images.append(recon + reconstructed_images[-1])
        # now downsample as necessary, so that these end up the same size as
        # their corresponding coefficients. We multiply by the factor of 4 here
        # in order to approximately equalize the steerable pyramid coefficient
        # values across scales. This could also be handled by making the
        # pyramid tight frame
        reconstructed_images[:-1] = [
            signal.shrink(r, 2 ** (self.n_scales - i)) * 4 ** (self.n_scales - i)
            for i, r in enumerate(reconstructed_images[:-1])
        ]
        return reconstructed_images

    def _compute_autocorr(
        self, mask: list[Tensor], coeffs_list: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Compute the autocorrelation of some statistics.

        Parameters
        ----------
        mask :
            The mask to use for weighting.
        coeffs_list :
            List (of length s) of tensors of shape (batch, channel, *, height,
            width), where * is zero or one additional dimensions. Intended use
            case: magnitude_pyr_coeffs (which is list of length n_scales of 5d
            tensors, with * containing n_orientations) or reconstructed_images
            (which is a list of length n_scales+1 of 4d tensors)

        Returns
        -------
        autocorrs :
            Tensor of shape (batch, channel, masks, n_autocorrs, *, s) containing the
            autocorrelation (up to distance ``spatial_corr_width//2``) of each element
            in ``coeffs_list``, computed independently over all but the final two
            dimensions. ``n_autocorrs`` is the number of unique autocorrelation values,
            which is approximately sptial_corr_width^2 / 2.
        vars :
            Tensor of shape (batch, channel, *masks, *, s) containing the variance of
            each element in ``coeffs_list``, computed independently over all but the
            final two dimensions. Note that by *masks, we indicate that the dimensions
            will not be combined, so that if ``len(masks)==2``, *masks would hold two
            dimensions.

        """
        if coeffs_list[0].ndim == 5:
            dims = "o"
            rolls_h = self._autocorr_rolls_h
            rolls_w = self._autocorr_rolls_w
            var_dim = -2
        elif coeffs_list[0].ndim == 4:
            dims = ""
            rolls_h = [r[:, :, 0] for r in self._autocorr_rolls_h]
            rolls_w = [r[:, :, 0] for r in self._autocorr_rolls_w]
            var_dim = -1
        else:
            raise ValueError(
                "coeffs_list must contain tensors of either 4 or 5 dimensions!"
            )
        autocorr_expr = (
            f"{self._mask_input_idx}, b c {dims} h w, "
            f"b c {dims} shift h w ->"
            f" b c {self._mask_output_idx} shift {dims}"
        )
        acs = []
        vars = []
        # iterate through scales
        for coeff, rolls_h, rolls_w, scale_mask in zip(
            coeffs_list, rolls_h, rolls_w, mask
        ):
            # the following two lines are equivalent to having two for loops over
            # range(-spatial_corr_width//2, spatial_corr_width//2) and using roll along
            # the last two indices, but is much more efficient, especially on the gpu.
            rolled_coeff = einops.repeat(
                coeff,
                f"b c {dims} h w -> b c {dims} shift h w",
                shift=self._n_autocorrs,
            )
            rolled_coeff = rolled_coeff.gather(-2, rolls_h).gather(-1, rolls_w)
            autocorr = einops.einsum(*scale_mask, coeff, rolled_coeff, autocorr_expr)
            # this returns a view of autocorr that just selects out the variance, while
            # preserving the number of dims. we have specifically placed the (0, 0)
            # shift, which corresponds to the variance, as the last element
            var = torch.narrow(autocorr, var_dim, -1, 1)
            # and then drop the variance from here
            acs.append(
                torch.narrow(autocorr, var_dim, 0, self._n_autocorrs - 1)
                / (var + self._stability_epsilon)
            )
            vars.append(var)
        acs = einops.rearrange(
            acs,
            (
                f"scales b c {self._mask_output_idx} shifts {dims} -> "
                f"b c ({self._mask_output_idx}) shifts {dims} scales"
            ),
        )
        vars = einops.rearrange(
            vars,
            (
                f"scales b c {self._mask_output_idx} shifts {dims} -> "
                f"b c {self._mask_output_idx} {dims} (shifts scales)"
            ),
            shifts=1,
        )
        return acs, vars

    def _compute_skew_kurtosis_recon(
        self, mask: list[Tensor], reconstructed_images: list[Tensor], var_recon: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute the skew and kurtosis of each lowpass reconstructed image.

        For each scale, if the ratio of its variance to the original image's
        pixel variance is below a threshold of
        torch.finfo(img_var.dtype).resolution (1e-6 for float32, 1e-15 for
        float64), skew and kurtosis are assigned default values of 0 or 3,
        respectively.

        Parameters
        ----------
        mask :
            The mask to use for weighting.
        reconstructed_images :
            List of length n_scales+1 containing the reconstructed unoriented
            image at each scale, from fine to coarse. The final image is
            reconstructed just from the residual lowpass image.
        var_recon :
            Tensor of shape (batch, channel, masks, n_scales+1) containing the
            variance of each tensor in reconstruced_images

        Returns
        -------
        skew_recon, kurtosis_recon :
            Tensors of shape (batch, channel, masks, n_scales+1) containing the skew
            and kurtosis, respectively, of each tensor in
            ``reconstructed_images``.

        """
        var_recon = einops.rearrange(
            var_recon,
            (
                f"b c {self._mask_output_idx} scales -> "
                f"b c ({self._mask_output_idx}) scales"
            ),
        )
        skew_recon = []
        kurtosis_recon = []
        for img, scale_mask in zip(reconstructed_images, mask):
            skew_recon.append(
                einops.einsum(
                    *scale_mask,
                    img.pow(3),
                    (
                        f"{self._mask_input_idx}, b c h w -> "
                        f"b c {self._mask_output_idx}"
                    ),
                )
            )
            kurtosis_recon.append(
                einops.einsum(
                    *scale_mask,
                    img.pow(4),
                    (
                        f"{self._mask_input_idx}, b c h w -> "
                        f"b c {self._mask_output_idx}"
                    ),
                )
            )
        skew_recon = einops.rearrange(
            skew_recon,
            (
                f"scales b c {self._mask_output_idx} ->"
                f" b c ({self._mask_output_idx}) scales"
            ),
        )
        kurtosis_recon = einops.rearrange(
            kurtosis_recon,
            (
                f"scales b c {self._mask_output_idx} -> "
                f"b c ({self._mask_output_idx}) scales"
            ),
        )
        skew_recon = skew_recon / (var_recon.pow(1.5) + self._stability_epsilon)
        kurtosis_recon = kurtosis_recon / (var_recon.pow(2) + self._stability_epsilon)
        return skew_recon, kurtosis_recon

    def _compute_cross_correlation(
        self,
        mask: list[Tensor],
        coeffs_tensor: list[Tensor],
        coeffs_tensor_other: list[Tensor],
        coeffs_var: Tensor | None = None,
        coeffs_other_var: Tensor | None = None,
    ) -> Tensor:
        """Compute cross-correlations.

        Parameters
        ----------
        coeffs_tensor, coeffs_tensor_other :
            The two lists of length scales, each containing 5d tensors of shape
            (batch, channel, n_orientations, height, width) to be correlated.
        coeffs_var, coeffs_other_var :
            Two optional tensors containing the variances of coeffs_tensor and
            coeffs_tensor_other, respectively, in case they've already been computed.
            Should be of shape (batch, channel, *masks, n_orientations, n_scales). Note
            that by *masks, we indicate that the dimensions should not be combined, so
            that if ``len(masks)==2``, *masks would hold two dimensions. Used to
            normalize the covariances into cross-correlations. Intended use is the
            output of ``_compute_autocorr``.

        Returns
        -------
        cross_corrs :
            Tensor of shape (batch, channel, masks, n_orientations, n_orientations,
            scales) containing the cross-correlations at each scale.

        """
        covars = []
        covar_expr = (
            f"{self._mask_input_idx}, b c o1 h w, b c o2 h w ->"
            f" b c {self._mask_output_idx} o1 o2"
        )
        var_expr = (
            f"{self._mask_input_idx}, b c o1 h w, b c o1 h w ->"
            f" b c {self._mask_output_idx} o1"
        )
        outer_prod_expr = (
            f"b c {self._mask_output_idx} o1, "
            f"b c {self._mask_output_idx} o2 ->"
            f" b c {self._mask_output_idx} o1 o2"
        )
        for i, (scale_mask, coeff, coeff_other) in enumerate(
            zip(mask, coeffs_tensor, coeffs_tensor_other)
        ):
            # compute the covariance
            covar = einops.einsum(*scale_mask, coeff, coeff_other, covar_expr)
            # Then normalize it to get the Pearson product-moment correlation
            # coefficient, see
            # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html.
            if coeffs_var is None:
                # First, compute the variances of each coeff
                coeff_var = einops.einsum(*scale_mask, coeff, coeff, var_expr)
            else:
                coeff_var = coeffs_var[..., i]
            if coeffs_other_var is None:
                # First, compute the variances of each coeff
                coeff_other_var = einops.einsum(
                    *scale_mask, coeff_other, coeff_other, var_expr
                )
            else:
                coeff_other_var = coeffs_other_var[..., i]
            # Then compute the outer product of those variances.
            var_outer_prod = einops.einsum(coeff_var, coeff_other_var, outer_prod_expr)
            # And the sqrt of this is what we use to normalize the covariance
            # into the cross-correlation
            std_outer_prod = (var_outer_prod + self._stability_epsilon).sqrt()
            covars.append(covar / (std_outer_prod + self._stability_epsilon))
        return einops.rearrange(
            covars,
            (
                f"scales b c {self._mask_output_idx} o1 o2 ->"
                f" b c ({self._mask_output_idx}) o1 o2 scales"
            ),
        )

    @staticmethod
    def _double_phase_pyr_coeffs(
        pyr_coeffs: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Upsample and double the phase of pyramid coefficients.

        Parameters
        ----------
        pyr_coeffs :
            Complex steerable pyramid coefficients (without residuals), as list
            of length n_scales, containing 5d tensors of shape (batch, channel,
            n_orientations, height, width)

        Returns
        -------
        doubled_phase_mags :
            The demeaned magnitude (i.e., pyr_coeffs.abs()) of each upsampled
            double-phased coefficient. List of length n_scales-1 containing
            tensors of same shape the input (the finest scale has been
            removed).
        doubled_phase_separate :
            The real and imaginary parts of each double-phased coefficient.
            List of length n_scales-1, containing tensors of shape (batch,
            channel, 2*n_orientations, height, width), with the real component
            found at the same orientation index as the input, and the imaginary
            at orientation+self.n_orientations. (The finest scale has been
            removed.)

        """
        doubled_phase_mags = []
        doubled_phase_sep = []
        # don't do this for the finest scale
        for coeff in pyr_coeffs[1:]:
            # We divide by the factor of 4 here in order to approximately
            # equalize the steerable pyramid coefficient values across scales.
            # This could also be handled by making the pyramid tight frame
            doubled_phase = signal.expand(coeff, 2) / 4.0
            doubled_phase = signal.modulate_phase(doubled_phase, 2)
            doubled_phase_mag = doubled_phase.abs()
            doubled_phase_mag = doubled_phase_mag - doubled_phase_mag.mean(
                (-2, -1), keepdim=True
            )
            doubled_phase_mags.append(doubled_phase_mag)
            doubled_phase_sep.append(
                einops.pack([doubled_phase.real, doubled_phase.imag], "b c * h w")[0]
            )
        return doubled_phase_mags, doubled_phase_sep

    def plot_representation(
        self,
        data: Tensor,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (15, 5),
        ylim: tuple[float, float] | Literal[False] | None = False,
        batch_idx: int = 0,
        title: str | None = None,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        r"""Plot the representation in a human viewable format -- stem
        plots with data separated out by statistic type.

        This plots the representation of a single batch and averages over all
        channels in the representation.

        We create the following axes:

        - pixels+var_highpass: marginal pixel statistics (first four moments,
          min, max) and variance of the residual highpass.

        - std+skew+kurtosis recon: the standard deviation, skew, and kurtosis
          of the reconstructed lowpass image at each scale

        - magnitude_std: the standard deviation of the steerable pyramid
          coefficient magnitudes at each orientation and scale.

        - auto_correlation_reconstructed: the auto-correlation of the
          reconstructed lowpass image at each scale (summarized using Euclidean
          norm).

        - auto_correlation_magnitude: the auto-correlation of the pyramid
          coefficient magnitudes at each scale and orientation (summarized
          using Euclidean norm).

        - cross_orientation_correlation_magnitude: the cross-correlations
          between each orientation at each scale (summarized using Euclidean
          norm)

        If self.n_scales > 1, we also have:

        - cross_scale_correlation_magnitude: the cross-correlations between the
          pyramid coefficient magnitude at one scale and the same orientation
          at the next-coarsest scale (summarized using Euclidean norm).

        - cross_scale_correlation_real: the cross-correlations between the real
          component of the pyramid coefficients and the real and imaginary
          components (at the same orientation) at the next-coarsest scale
          (summarized using Euclidean norm).

        Parameters
        ----------
        data :
            The data to show on the plot. Else, should look like the output of
            ``self.forward(img)``, with the exact same structure (e.g., as
            returned by ``metamer.representation_error()`` or another instance
            of this class).
        ax :
            Axes where we will plot the data. If a ``plt.Axes`` instance, will
            subdivide into 6 or 8 new axes (depending on self.n_scales). If
            None, we create a new figure.
        figsize :
            The size of the figure. Ignored if ax is not None.
        ylim :
            If not None, the y-limits to use for this plot. If None, we use the
            default, slightly adjusted so that the minimum is 0. If False, do not
            change y-limits.
        batch_idx :
            Which index to take from the batch dimension (the first one)
        title : string
            Title for the plot

        Returns
        -------
        fig:
            Figure containing the plot
        axes:
            List of 6 or 8 axes containing the plot (depending on self.n_scales)

        """
        if self.n_scales != 1:
            n_rows = 3
            n_cols = 3
        else:
            # then we don't have any cross-scale correlations, so fewer axes.
            n_rows = 2
            n_cols = 3

        # pick the batch_idx we want (but keep the data 3d), and average over
        # channels (but keep the data 3d). We keep data 3d because
        # convert_to_dict relies on it.
        data = data[batch_idx].unsqueeze(0).mean(1, keepdim=True)
        # each of these values should now be a 3d tensor with 1 element in each
        # of the first two dims
        rep = {k: v[0, 0] for k, v in self.convert_to_dict(data).items()}
        data = self._representation_for_plotting(rep)

        # Set up grid spec
        if ax is None:
            # we add 2 to order because we're adding one to get the
            # number of orientations and then another one to add an
            # extra column for the mean luminance plot
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(n_rows, n_cols, fig)
        else:
            # warnings.warn("ax is not None, so we're ignoring figsize...")
            # want to make sure the axis we're taking over is basically invisible.
            ax = clean_up_axes(
                ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
            )
            gs = ax.get_subplotspec().subgridspec(n_rows, n_cols)
            fig = ax.figure

        # plot data
        axes = []
        for i, (k, v) in enumerate(data.items()):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax = clean_stem_plot(to_numpy(v).flatten(), ax, k, ylim=ylim)
            axes.append(ax)

        if title is not None:
            fig.suptitle(title)

        return fig, axes

    def _representation_for_plotting(self, rep: OrderedDict) -> OrderedDict:
        r"""Convert the data into a more convenient representation for plotting.

        Intended as a helper function for plot_representation.

        """
        data = OrderedDict()
        data["pixels+var_highpass"] = torch.cat(
            [rep.pop("pixel_statistics"), rep.pop("var_highpass_residual")], -1
        )
        data["std+skew+kurtosis recon"] = torch.cat(
            (
                rep.pop("std_reconstructed"),
                rep.pop("skew_reconstructed"),
                rep.pop("kurtosis_reconstructed"),
            ),
            -1,
        )

        data["magnitude_std"] = rep.pop("magnitude_std").flatten(1)

        # want to plot these in a specific order
        all_keys = [
            "auto_correlation_reconstructed",
            "auto_correlation_magnitude",
            "cross_orientation_correlation_magnitude",
            "cross_scale_correlation_magnitude",
            "cross_scale_correlation_real",
        ]
        if set(rep.keys()) != set(all_keys):
            raise ValueError("representation has unexpected keys!")
        for k in all_keys:
            # if we only have one scale, no cross-scale stats
            if k.startswith("cross_scale") and self.n_scales == 1:
                continue
            # these will then be 2d, with masks on the first dimension
            if k == "cross_orientation_correlation_magnitude":
                # this one has nans in it (indicating unnecessary stats), and so we need
                # to compute the L2 norm ourselves
                data[k] = rep[k].pow(2).nansum((1, 2)).sqrt()
            else:
                data[k] = torch.linalg.vector_norm(rep[k], ord=2, dim=1).flatten(1)

        return data

    def update_plot(
        self,
        axes: list[plt.Axes],
        data: Tensor,
        batch_idx: int = 0,
    ) -> list[plt.Artist]:
        r"""Update the information in our representation plot.

        This is used for creating an animation of the representation
        over time. In order to create the animation, we need to know how
        to update the matplotlib Artists, and this provides a simple way
        of doing that. It relies on the fact that we've used
        ``plot_representation`` to create the plots we want to update
        and so know that they're stem plots.

        We take the axes containing the representation information (note that
        this is probably a subset of the total number of axes in the figure, if
        we're showing other information, as done by ``Metamer.animate``), grab
        the representation from plotting and, since these are both lists,
        iterate through them, updating them to the values in ``data`` as we go.

        In order for this to be used by ``FuncAnimation``, we need to
        return Artists, so we return a list of the relevant artists, the
        ``markerline`` and ``stemlines`` from the ``StemContainer``.

        Currently, this averages over all channels in the representation.

        Parameters
        ----------
        axes :
            A list of axes to update. We assume that these are the axes
            created by ``plot_representation`` and so contain stem plots
            in the correct order.
        batch_idx :
            Which index to take from the batch dimension (the first one)
        data :
            The data to show on the plot. Else, should look like the output of
            ``self.forward(img)``, with the exact same structure (e.g., as
            returned by ``metamer.representation_error()`` or another instance
            of this class).

        Returns
        -------
        stem_artists :
            A list of the artists used to update the information on the
            stem plots

        """
        stem_artists = []
        axes = [ax for ax in axes if len(ax.containers) == 1]
        # pick the batch_idx we want (but keep the data 3d), and average over
        # channels (but keep the data 3d). We keep data 3d because
        # convert_to_dict relies on it.
        data = data[batch_idx].unsqueeze(0).mean(1, keepdim=True)
        # each of these values should now be a 3d tensor with 1 element in each
        # of the first two dims
        rep = {k: v[0, 0] for k, v in self.convert_to_dict(data).items()}
        rep = self._representation_for_plotting(rep)
        for ax, d in zip(axes, rep.values()):
            vals = to_numpy(d.flatten())
            sc = update_stem(ax.containers[0], vals)
            stem_artists.extend([sc.markerline, sc.stemlines])
        return stem_artists
