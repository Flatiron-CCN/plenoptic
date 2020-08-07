import numpy as np
import torch
import torch.nn.functional as F

from ..simulate.canonical_computations import Laplacian_Pyramid, Steerable_Pyramid_Freq
from ..simulate.canonical_computations import local_gain_control, rectangular_to_polar_dict

import os
dirname = os.path.dirname(__file__)

# TODO: clean up, test and document (MS)SSIM

def gaussian(window_size, sigma):
    gauss = torch.tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    """Structural similarity index

    As described in  [1]_,

    Argument
    --------
    img1:
    img2:
    window_size:
    window:
    size_average:
    full:
        contrast sensitivity
    val_range:
        Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    Return
    ------
    ssim
    cs
    ssim_map: TODO

    References
    ----------
    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error measurement to structural similarity" IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
    .. [3] [project page](https://www.cns.nyu.edu/~lcv/ssim/)
    .. [2] [matlab code](https://www.cns.nyu.edu/~lcv/ssim/ssim_index.m)
    """

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def normalized_laplacian_pyramid(im):
    """computes the normalized Laplacian Pyramid using pre-optimized parameters

    Arguments
    --------
    im: torch.Tensor
    Returns
    -------
    normalized_laplacian_activations: list of torch.Tensor
    """

    (_, channel, height, width) = im.size()

    N_scales = 6
    spatialpooling_filters = np.load(dirname + '/DN_filts.npy')
    sigmas = np.load(dirname + '/DN_sigmas.npy')

    L = Laplacian_Pyramid(n_scales=N_scales)
    laplacian_activations = L.analysis(im)

    padd = 2
    normalized_laplacian_activations = []
    for N_b in range(0, N_scales):
        filt = torch.tensor(spatialpooling_filters[N_b], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        filtered_activations = F.conv2d(torch.abs(laplacian_activations[N_b]), filt, padding=padd, groups=channel)
        normalized_laplacian_activations.append(laplacian_activations[N_b] / (sigmas[N_b] + filtered_activations))

    return normalized_laplacian_activations


def nlpd(IM_1, IM_2):
    """Normalized Laplacian Pyramid Distance

    As described in  [1]_, this is an image quality metric based on the transformations associated with the early
    visual system: local luminance subtraction and local contrast gain control

    A laplacian pyramid subtracts a local estimate of the mean luminance at six scales.
    Then a local gain control divides these centered coefficients by a weighted sum of absolute values
    in spatial neighborhood.

    These weights parameters were optimized for redundancy reduction over an training
    database of (undistorted) natural images.

    Note that we compute root mean squared error for each scale, and then average over these,
    effectively giving larger weight to the lower frequency coefficients
    (which are fewer in number, due to subsampling).

    Parameters
    ----------
    IM_1: torch.Tensor
        image, (1 x 1 x H x W)
    IM_2: torch.Tensor
        image, (1 x 1 x H x W)

    Returns
    -------
    distance: float

    Note
    ----
    only accepts single channel images

    References
    ----------
    .. [1] Laparra, V., Ballé, J., Berardino, A. and Simoncelli, E.P., 2016. Perceptual image quality assessment using a normalized Laplacian pyramid. Electronic Imaging, 2016(16), pp.1-6.
    """

    y = normalized_laplacian_pyramid(torch.cat((IM_1, IM_2), 0))

    # for optimization purpose (stabilizing the gradient around zero)
    epsilon = 1e-10
    dist = []
    for i in range(6):
        dist.append(torch.sqrt(torch.mean((y[i][0] - y[i][1]) ** 2) + epsilon))

    return torch.stack(dist).mean()


def nspd(IM_1, IM_2, O=1, S=5, complex=True):
    """Normalized steerable pyramid distance

    spatially local normalization pool

    under construction
    """

    if complex:
        linear = Steerable_Pyramid_Freq(IM_1.shape[-2:], order=O, height=S, is_complex=True)
        non_linear = rectangular_to_polar_dict
    else:
        linear = Steerable_Pyramid_Freq(IM_1.shape[-2:], order=O, height=S)
        non_linear = local_gain_control

    pyr = linear(torch.cat((IM_1, IM_2), 0))

    norm, state = non_linear(pyr)

    # for optimization purpose (stabilizing the gradient around zero)
    epsilon = 1e-10
    dist = []
    for key in state.keys():
        # TODO learn weights on TID2013
        dist.append(torch.sqrt(torch.mean((norm[key][0] - norm[key][1]) ** 2) + epsilon))
        dist.append(torch.sqrt(torch.mean((state[key][0] - state[key][1]) ** 2) + epsilon))

    return torch.stack(dist).mean()


# # TODO: Spectral Residual based Similarity
# function sim = SR_SIM(image1, image2)
# % ========================================================================
# % SR_SIM Index with automatic downsampling, Version 1.0
# % Copyright(c) 2011 Lin ZHANG
# % All Rights Reserved.
# %
# % ----------------------------------------------------------------------
# % Permission to use, copy, or modify this software and its documentation
# % for educational and research purposes only and without fee is hereQ
# % granted, provided that this copyright notice and the original authors'
# % names appear on all copies and supporting documentation. This program
# % shall not be used, rewritten, or adapted as the basis of a commercial
# % software or hardware product without first obtaining permission of the
# % authors. The authors make no representations about the suitability of
# % this software for any purpose. It is provided "as is" without express
# % or implied warranty.
# %----------------------------------------------------------------------
# %
# % This is an implementation of the algorithm for calculating the
# % Spectral Residual based Similarity (SR-SIM) index between two images. For
# % more details, please refer to our paper:
# % Lin Zhang and Hongyu Li, "SR-SIM: A fast and high performance IQA index based on spectral residual", in: Proc. ICIP 2012.
# %
# %----------------------------------------------------------------------
# %
# %Input : (1) image1: the first image being compared
# %        (2) image2: the second image being compared
# %
# %Output: sim: the similarity score between two images, a real number
# %
# %-----------------------------------------------------------------------
# [rows, cols, junk] = size(image1);
# if junk == 3
#     Y1 = 0.299 * double(image1(:,:,1)) + 0.587 * double(image1(:,:,2)) + 0.114 * double(image1(:,:,3));
#     Y2 = 0.299 * double(image2(:,:,1)) + 0.587 * double(image2(:,:,2)) + 0.114 * double(image2(:,:,3));
# else
#     Y1 = double(image1);
#     Y2 = double(image2);
# end
#
# %%%%%%%%%%%%%%%%%%%%%%%%%
# % Download the image
# %%%%%%%%%%%%%%%%%%%%%%%%%
# minDimension = min(rows,cols);
# F = max(1,round(minDimension / 256));
# aveKernel = fspecial('average',F);
#
# aveY1 = conv2(Y1, aveKernel,'same');
# aveY2 = conv2(Y2, aveKernel,'same');
# Y1 = aveY1(1:F:rows,1:F:cols);
# Y2 = aveY2(1:F:rows,1:F:cols);
#
# %%%%%%%%%%%%%%%%%%%%%%%%%
# % Calculate the visual saliency maps
# %%%%%%%%%%%%%%%%%%%%%%%%%
# saliencyMap1 = spectralResidueSaliency(Y1);
# saliencyMap2 = spectralResidueSaliency(Y2);
# %%%%%%%%%%%%%%%%%%%%%%%%%
# % Calculate the gradient map
# %%%%%%%%%%%%%%%%%%%%%%%%%
# dx = [3 0 -3; 10 0 -10;  3  0 -3]/16;
# dy = [3 10 3; 0  0   0; -3 -10 -3]/16;
# IxY1 = conv2(Y1, dx, 'same');
# IyY1 = conv2(Y1, dy, 'same');
# gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);
#
# IxY2 = conv2(Y2, dx, 'same');
# IyY2 = conv2(Y2, dy, 'same');
# gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);
#
# %%%%%%%%%%%%%%%%%%%%%%%%%
# % Calculate the SR-SIM
# %%%%%%%%%%%%%%%%%%%%%%%%%
# C1 = 0.40; %fixed
# C2 = 225;
# alpha = 0.50;%fixed
#
# GBVSSimMatrix = (2 * saliencyMap1 .* saliencyMap2 + C1) ./ (saliencyMap1.^2 + saliencyMap2.^2 + C1);
# gradientSimMatrix = (2*gradientMap1.*gradientMap2 + C2) ./(gradientMap1.^2 + gradientMap2.^2 + C2);
#
# weight = max(saliencyMap1, saliencyMap2);
# SimMatrix = GBVSSimMatrix .* (gradientSimMatrix .^ alpha) .* weight;
# sim = sum(sum(SimMatrix)) / sum(weight(:));
#
# return;
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function saliencyMap = spectralResidueSaliency(image)
# %this function is used to calculate the visual saliency map for the given
# %image using the spectral residue method proposed by Xiaodi Hou and Liqing
# %Zhang. For more details about this method, you can refer to the paper:
# %Saliency detection: a spectral residual approach.
#
# %there are some parameters needed to be adjusted
# scale = 0.25; %fixed
# aveKernelSize = 3; %fixed
# gauSigma = 3.8; %fixed
# gauSize = 10; %fixed
#
# inImg = imresize(image, scale);
#
# %%%% Spectral Residual
# myFFT = fft2(inImg);
# myLogAmplitude = log(abs(myFFT));
# myPhase = angle(myFFT);
#
# mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', aveKernelSize), 'replicate');
# saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;
#
# %%%% After Effect
# saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [gauSize, gauSize], gauSigma)));
# saliencyMap = imresize(saliencyMap,[size(image,1) size(image,2)]);
