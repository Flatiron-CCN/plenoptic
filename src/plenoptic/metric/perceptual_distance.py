import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..simulate.canonical_computations import LaplacianPyramid
from ..simulate.canonical_computations.filters import circular_gaussian2d
from ..tools.conv import same_padding

DIRNAME = resources.files("plenoptic.metric")


def _ssim_parts(img1, img2, pad=False):
    """Calcluates the various components used to compute SSIM

    This should not be called by users directly, but is meant to assist for
    calculating SSIM and MS-SSIM.

    Parameters
    ----------
    img1: torch.Tensor of shape (batch, channel, height, width)
        The first image or batch of images.
    img2: torch.Tensor of shape (batch, channel, height, width)
        The second image or batch of images. The heights and widths of `img1`
        and `img2` must be the same. The numbers of batches and channels of
        `img1` and `img2` need to be broadcastable: either they are the same
        or one of them is 1. The output will be computed separately for each
        channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may
        be inaccurate, and we will raise a warning (but will still compute it).
    pad : {False, 'constant', 'reflect', 'replicate', 'circular'}, optional
        If not False, how to pad the image for the convolutions computing the
        local average of each image. See `torch.nn.functional.pad` for how
        these work.

    """
    img_ranges = torch.as_tensor([[img1.min(), img1.max()], [img2.min(), img2.max()]])
    if (img_ranges > 1).any() or (img_ranges < 0).any():
        warnings.warn(
            "Image range falls outside [0, 1]."
            f" img1: {img_ranges[0]}, img2: {img_ranges[1]}. "
            "Continuing anyway..."
        )

    if not img1.ndim == img2.ndim == 4:
        raise Exception(
            "Input images should have four dimensions: (batch, channel, height, width)"
        )
    if img1.shape[-2:] != img2.shape[-2:]:
        raise Exception("img1 and img2 must have the same height and width!")
    for i in range(2):
        if img1.shape[i] != img2.shape[i] and img1.shape[i] != 1 and img2.shape[i] != 1:
            raise Exception(
                "Either img1 and img2 should have the same number of "
                "elements in each dimension, or one of "
                "them should be 1! But got shapes "
                f"{img1.shape}, {img2.shape} instead"
            )
    if img1.shape[1] > 1 or img2.shape[1] > 1:
        warnings.warn(
            "SSIM was designed for grayscale images and here it will be"
            " computed separately for each channel (so channels are treated in"
            " the same way as batches)."
        )
    if img1.dtype != img2.dtype:
        raise ValueError("Input images must have same dtype!")

    real_size = min(11, img1.shape[2], img1.shape[3])
    std = torch.as_tensor(1.5).to(img1.device)
    window = circular_gaussian2d(real_size, std=std).to(img1.dtype)

    # these two checks are guaranteed with our above bits, but if we add
    # ability for users to set own window, they'll be necessary
    window_sum = window.sum((-1, -2), keepdim=True)
    if not torch.allclose(window_sum, torch.ones_like(window_sum)):
        warnings.warn("window should have sum of 1! normalizing...")
        window = window / window_sum
    if window.ndim != 4:
        raise Exception("window must have 4 dimensions!")

    if pad is not False:
        img1 = same_padding(img1, (real_size, real_size), pad_mode=pad)
        img2 = same_padding(img2, (real_size, real_size), pad_mode=pad)

    def windowed_average(img):
        padd = 0
        (n_batches, n_channels, _, _) = img.shape
        img = img.reshape(n_batches * n_channels, 1, img.shape[2], img.shape[3])
        img_average = F.conv2d(img, window, padding=padd)
        img_average = img_average.reshape(
            n_batches, n_channels, img_average.shape[2], img_average.shape[3]
        )
        return img_average

    mu1 = windowed_average(img1)
    mu2 = windowed_average(img2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = windowed_average(img1 * img1) - mu1_sq
    sigma2_sq = windowed_average(img2 * img2) - mu2_sq
    sigma12 = windowed_average(img1 * img2) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    # SSIM is the product of a luminance component, a contrast component, and a
    # structure component. The contrast-structure component has to be separated
    # when computing MS-SSIM.
    luminance_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast_structure_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    map_ssim = luminance_map * contrast_structure_map

    # the weight used for stability
    weight = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
    return map_ssim, contrast_structure_map, weight


def ssim(img1, img2, weighted=False, pad=False):
    r"""Structural similarity index

    As described in [1]_, the structural similarity index (SSIM) is a
    perceptual distance metric, giving the distance between two images. SSIM is
    based on three comparison measurements between the two images: luminance,
    contrast, and structure. All of these are computed convolutionally across the
    images. See the references for more information.

    This implementation follows the original implementation, as found at [2]_,
    as well as providing the option to use the weighted version used in [4]_
    (which was shown to consistently improve the image quality prediction on
    the LIVE database).

    Note that this is a similarity metric (not a distance), and so 1 means the
    two images are identical and 0 means they're very different. When the two
    images are negatively correlated, SSIM can be negative. SSIM is bounded
    between -1 and 1.

    This function returns the mean SSIM, a scalar-valued metric giving the
    average over the whole image. For the SSIM map (showing the computed value
    across the image), call `ssim_map`.

    Parameters
    ----------
    img1: torch.Tensor of shape (batch, channel, height, width)
        The first image or batch of images.
    img2: torch.Tensor of shape (batch, channel, height, width)
        The second image or batch of images. The heights and widths of `img1`
        and `img2` must be the same. The numbers of batches and channels of
        `img1` and `img2` need to be broadcastable: either they are the same
        or one of them is 1. The output will be computed separately for each
        channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may
        be inaccurate, and we will raise a warning (but will still compute it).
    weighted : bool, optional
        whether to use the original, unweighted SSIM version (`False`) as used
        in [1]_ or the weighted version (`True`) as used in [4]_. See Notes
        section for the weight
    pad : {False, 'constant', 'reflect', 'replicate', 'circular'}, optional
        If not False, how to pad the image for the convolutions computing the
        local average of each image. See `torch.nn.functional.pad` for how
        these work.

    Returns
    -------
    mssim : torch.Tensor
        2d tensor of shape (batch, channel) containing the mean SSIM for each
        image, averaged over the whole image

    Notes
    -----
    The weight used when `weighted=True` is:

    .. math::
       \log((1+\frac{\sigma_1^2}{C_2})(1+\frac{\sigma_2^2}{C_2}))

    where :math:`sigma_1^2` and :math:`sigma_2^2` are the variances of `img1`
    and `img2`, respectively, and :math:`C_2` is a constant. See [4]_ for more
    details.

    References
    ----------
    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
       quality assessment: From error measurement to structural similarity"
       IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [2] [matlab code](https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m)
    .. [3] [project page](https://www.cns.nyu.edu/~lcv/ssim/)
    .. [4] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of
       perceptual discriminability. Journal of Vision, 8(12), 1–13.
       https://dx.doi.org/10.1167/8.12.8

    """
    # these are named map_ssim instead of the perhaps more natural ssim_map
    # because that's the name of a function
    map_ssim, _, weight = _ssim_parts(img1, img2, pad)
    if not weighted:
        mssim = map_ssim.mean((-1, -2))
    else:
        mssim = (map_ssim * weight).sum((-1, -2)) / weight.sum((-1, -2))

    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn(
            "SSIM uses 11x11 convolutional kernel, but the height and/or "
            "the width of the input image is smaller than 11, so the "
            "kernel size is set to be the minimum of these two numbers."
        )
    return mssim


def ssim_map(img1, img2):
    """Structural similarity index map

    As described in [1]_, the structural similarity index (SSIM) is a
    perceptual distance metric, giving the distance between two images. SSIM is
    based on three comparison measurements between the two images: luminance,
    contrast, and structure. All of these are computed convolutionally across the
    images. See the references for more information.

    This implementation follows the original implementation, as found at [2]_,
    as well as providing the option to use the weighted version used in [4]_
    (which was shown to consistently improve the image quality prediction on
    the LIVE database).

    Note that this is a similarity metric (not a distance), and so 1 means the
    two images are identical and 0 means they're very different. When the two
    images are negatively correlated, SSIM can be negative. SSIM is bounded
    between -1 and 1.

    This function returns the SSIM map, showing the SSIM values across the
    image. For the mean SSIM (a single value metric), call `ssim`.

    Parameters
    ----------
    img1: torch.Tensor of shape (batch, channel, height, width)
        The first image or batch of images.
    img2: torch.Tensor of shape (batch, channel, height, width)
        The second image or batch of images. The heights and widths of `img1`
        and `img2` must be the same. The numbers of batches and channels of
        `img1` and `img2` need to be broadcastable: either they are the same
        or one of them is 1. The output will be computed separately for each
        channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may
        be inaccurate, and we will raise a warning (but will still compute it).
    weighted : bool, optional
        whether to use the original, unweighted SSIM version (`False`) as used
        in [1]_ or the weighted version (`True`) as used in [4]_. See Notes
        section for the weight

    Returns
    -------
    ssim_map : torch.Tensor
        4d tensor containing the map of SSIM values.

    References
    ----------
    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
       quality assessment: From error measurement to structural similarity"
       IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [2] [matlab code](https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m)
    .. [3] [project page](https://www.cns.nyu.edu/~lcv/ssim/)
    .. [4] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of
       perceptual discriminability. Journal of Vision, 8(12), 1–13.
       https://dx.doi.org/10.1167/8.12.8

    """
    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn(
            "SSIM uses 11x11 convolutional kernel, but the height and/or "
            "the width of the input image is smaller than 11, so the "
            "kernel size is set to be the minimum of these two numbers."
        )
    return _ssim_parts(img1, img2)[0]


def ms_ssim(img1, img2, power_factors=None):
    r"""Multiscale structural similarity index (MS-SSIM)

    As described in [1]_, multiscale structural similarity index (MS-SSIM) is
    an improvement upon structural similarity index (SSIM) that takes into
    account the perceptual distance between two images on different scales.

    SSIM is based on three comparison measurements between the two images:
    luminance, contrast, and structure. All of these are computed convolutionally
    across the images, producing three maps instead of scalars. The SSIM map is
    the elementwise product of these three maps. See `metric.ssim` and
    `metric.ssim_map` for a full description of SSIM.

    To get images of different scales, average pooling operations with kernel
    size 2 are performed recursively on the input images. The product of
    contrast map and structure map (the "contrast-structure map") is computed
    for all but the coarsest scales, and the overall SSIM map is only computed
    for the coarsest scale. Their mean values are raised to exponents and
    multiplied to produce MS-SSIM:

    .. math::
        MSSSIM = {SSIM}_M^{a_M} \prod_{i=1}^{M-1} ({CS}_i)^{a_i}

    Here :math: `M` is the number of scales, :math: `{CS}_i` is the mean value
    of the contrast-structure map for the i'th finest scale, and :math: `{SSIM}_M`
    is the mean value of the SSIM map for the coarsest scale. If at least one
    of these terms are negative, the value of MS-SSIM is zero. The values of
    :math: `a_i, i=1,...,M` are taken from the argument `power_factors`.

    Parameters
    ----------
    img1: torch.Tensor of shape (batch, channel, height, width)
        The first image or batch of images.
    img2: torch.Tensor of shape (batch, channel, height, width)
        The second image or batch of images. The heights and widths of `img1`
        and `img2` must be the same. The numbers of batches and channels of
        `img1` and `img2` need to be broadcastable: either they are the same
        or one of them is 1. The output will be computed separately for each
        channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may
        be inaccurate, and we will raise a warning (but will still compute it).
    power_factors : 1D array, optional.
        power exponents for the mean values of maps, for different scales (from
        fine to coarse). The length of this array determines the number of scales.
        By default, this is set to [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        which is what psychophysical experiments in [1]_ found.

    Returns
    -------
    msssim : torch.Tensor
        2d tensor of shape (batch, channel) containing the MS-SSIM for each image

    References
    ----------
    .. [1] Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
       structural similarity for image quality assessment." The Thrity-Seventh
       Asilomar Conference on Signals, Systems & Computers, 2003. Vol. 2. IEEE, 2003.

    """
    if power_factors is None:
        power_factors = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def downsample(img):
        img = F.pad(img, (0, img.shape[3] % 2, 0, img.shape[2] % 2), mode="replicate")
        img = F.avg_pool2d(img, kernel_size=2)
        return img

    msssim = 1
    for i in range(len(power_factors) - 1):
        _, contrast_structure_map, _ = _ssim_parts(img1, img2)
        msssim *= F.relu(contrast_structure_map.mean((-1, -2))).pow(power_factors[i])
        img1 = downsample(img1)
        img2 = downsample(img2)
    map_ssim, _, _ = _ssim_parts(img1, img2)
    msssim *= F.relu(map_ssim.mean((-1, -2))).pow(power_factors[-1])

    if min(img1.shape[2], img1.shape[3]) < 11:
        warnings.warn(
            "SSIM uses 11x11 convolutional kernel, but for some scales "
            "of the input image, the height and/or the width is smaller "
            "than 11, so the kernel size in SSIM is set to be the "
            "minimum of these two numbers for these scales."
        )
    return msssim


def normalized_laplacian_pyramid(img):
    """Compute the normalized Laplacian Pyramid using pre-optimized parameters

    Parameters
    ----------
    img: torch.Tensor of shape (batch, channel, height, width)
        Image, or batch of images. This representation is designed
        for grayscale images and will be computed separately for each
        channel (so channels are treated in the same way as batches).

    Returns
    -------
    normalized_laplacian_activations: list of torch.Tensor
        The normalized Laplacian Pyramid with six scales
    """

    (_, channel, height, width) = img.size()

    N_scales = 6
    spatialpooling_filters = np.load(Path(DIRNAME) / "DN_filts.npy")

    sigmas = np.load(Path(DIRNAME) / "DN_sigmas.npy")

    L = LaplacianPyramid(n_scales=N_scales, scale_filter=True)
    laplacian_activations = L.forward(img)

    padd = 2
    normalized_laplacian_activations = []
    for N_b in range(0, N_scales):
        filt = torch.as_tensor(
            spatialpooling_filters[N_b], dtype=img.dtype, device=img.device
        ).repeat(channel, 1, 1, 1)
        filtered_activations = F.conv2d(
            torch.abs(laplacian_activations[N_b]),
            filt,
            padding=padd,
            groups=channel,
        )
        normalized_laplacian_activations.append(
            laplacian_activations[N_b] / (sigmas[N_b] + filtered_activations)
        )

    return normalized_laplacian_activations


def nlpd(img1, img2):
    """Normalized Laplacian Pyramid Distance

    As described in  [1]_, this is an image quality metric based on the transformations
    associated with the early visual system: local luminance subtraction and local
    contrast gain control.

    A laplacian pyramid subtracts a local estimate of the mean luminance at six scales.
    Then a local gain control divides these centered coefficients by a weighted sum of
    absolute values in spatial neighborhood.

    These weights parameters were optimized for redundancy reduction over an training
    database of (undistorted) natural images.

    Note that we compute root mean squared error for each scale, and then average over
    these, effectively giving larger weight to the lower frequency coefficients
    (which are fewer in number, due to subsampling).

    Parameters
    ----------
    img1: torch.Tensor of shape (batch, channel, height, width)
        The first image or batch of images.
    img2: torch.Tensor of shape (batch, channel, height, width)
        The second image or batch of images. The heights and widths of `img1`
        and `img2` must be the same. The numbers of batches and channels of
        `img1` and `img2` need to be broadcastable: either they are the same
        or one of them is 1. The output will be computed separately for each
        channel (so channels are treated in the same way as batches). Both
        images should have values between 0 and 1. Otherwise, the result may
        be inaccurate, and we will raise a warning (but will still compute it).

    Returns
    -------
    distance: torch.Tensor of shape (batch, channel)
        The normalized Laplacian Pyramid distance.

    References
    ----------
    .. [1] Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual
        image quality assessment using a normalized Laplacian pyramid. Electronic
        Imaging, 2016(16), pp.1-6.
    """

    if not img1.ndim == img2.ndim == 4:
        raise Exception(
            "Input images should have four dimensions: (batch, channel, height, width)"
        )
    if img1.shape[-2:] != img2.shape[-2:]:
        raise Exception("img1 and img2 must have the same height and width!")
    for i in range(2):
        if img1.shape[i] != img2.shape[i] and img1.shape[i] != 1 and img2.shape[i] != 1:
            raise Exception(
                "Either img1 and img2 should have the same number of "
                "elements in each dimension, or one of "
                "them should be 1! But got shapes "
                f"{img1.shape}, {img2.shape} instead"
            )
    if img1.shape[1] > 1 or img2.shape[1] > 1:
        warnings.warn(
            "NLPD was designed for grayscale images and here it will be"
            " computed separately for each channel (so channels are treated in"
            " the same way as batches)."
        )

    img_ranges = torch.as_tensor([[img1.min(), img1.max()], [img2.min(), img2.max()]])
    if (img_ranges > 1).any() or (img_ranges < 0).any():
        warnings.warn(
            "Image range falls outside [0, 1]."
            f" img1: {img_ranges[0]}, img2: {img_ranges[1]}. "
            "Continuing anyway..."
        )

    y1 = normalized_laplacian_pyramid(img1)
    y2 = normalized_laplacian_pyramid(img2)

    epsilon = 1e-10  # for optimization purpose (stabilizing the gradient around zero)
    dist = []
    for i in range(6):
        dist.append(torch.sqrt(torch.mean((y1[i] - y2[i]) ** 2, dim=(2, 3)) + epsilon))

    return torch.stack(dist).mean(dim=0)
