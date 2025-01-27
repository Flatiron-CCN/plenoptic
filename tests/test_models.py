# we do this to enable deterministic behavior on the gpu, see
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility for
# details
import os
from collections import OrderedDict
from contextlib import nullcontext as does_not_raise

import einops
import matplotlib.pyplot as plt
import numpy as np
import pyrtools as pt
import pytest
import scipy.io as sio
import torch

import plenoptic as po
from conftest import DEVICE, IMG_DIR
from plenoptic.simulate.canonical_computations import circular_gaussian2d, gaussian1d

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


ALL_MODELS = [
    "LPyr",
    "SPyr",
    "frontend.LinearNonlinear",
    "frontend.LuminanceGainControl",
    "frontend.LuminanceContrastGainControl",
    "frontend.OnOff",
    "naive.Identity",
    "naive.Linear",
    "naive.Gaussian",
    "naive.CenterSurround",
    "PortillaSimoncelli",
    "Identity",
    "NLP",
]


@pytest.fixture()
def portilla_simoncelli_matlab_test_vectors():
    return po.data.fetch_data("portilla_simoncelli_matlab_test_vectors.tar.gz")


@pytest.fixture()
def portilla_simoncelli_test_vectors():
    return po.data.fetch_data("portilla_simoncelli_test_vectors_refactor.tar.gz")


@pytest.fixture()
def portilla_simoncelli_synthesize():
    return po.data.fetch_data(
        "portilla_simoncelli_synthesize_torch_v1.12.0_ps-refactor-2.npz"
    )


@pytest.fixture()
def portilla_simoncelli_scales():
    # During PS refactor, we changed the structure of the
    # _representation_scales attribute, so have a different file to test
    # against
    return po.data.fetch_data("portilla_simoncelli_scales_ps-refactor.npz")


@pytest.mark.parametrize("model", ALL_MODELS, indirect=True)
@pytest.mark.skipif(DEVICE.type == "cpu", reason="Can only test on cuda")
def test_cuda(model, einstein_img):
    model.cuda()
    model(einstein_img)
    # make sure it ends on same device it started, since it might be a fixture
    model.to(DEVICE)


@pytest.mark.parametrize("model", ALL_MODELS, indirect=True)
@pytest.mark.skipif(DEVICE.type == "cpu", reason="Can only test on cuda")
def test_cpu_and_back(model, einstein_img):
    model.cpu()
    model.cuda()
    model(einstein_img)
    # make sure it ends on same device it started, since it might be a fixture
    model.to(DEVICE)


@pytest.mark.parametrize("model", ALL_MODELS, indirect=True)
@pytest.mark.skipif(DEVICE.type == "cpu", reason="Can only test on cuda")
def test_cuda_and_back(model, einstein_img):
    model.cuda()
    model.cpu()
    model(einstein_img.cpu())
    # make sure it ends on same device it started, since it might be a fixture
    einstein_img.to(DEVICE)
    model.to(DEVICE)


@pytest.mark.parametrize("model", ALL_MODELS, indirect=True)
def test_cpu(model, einstein_img):
    model.cpu()
    model(einstein_img.cpu())
    # make sure it ends on same device it started, since it might be a fixture
    einstein_img.to(DEVICE)
    model.to(DEVICE)


@pytest.mark.parametrize("model", ALL_MODELS, indirect=True)
def test_validate_model(model):
    po.tools.remove_grad(model)
    po.tools.validate.validate_model(model, device=DEVICE, image_shape=(1, 1, 256, 256))


class TestNonLinearities:
    def test_rectangular_to_polar_dict(self, basic_stim):
        spc = po.simul.SteerablePyramidFreq(
            basic_stim.shape[-2:],
            height=5,
            order=1,
            is_complex=True,
            tight_frame=True,
        ).to(DEVICE)
        y = spc(basic_stim)
        energy, state = po.simul.non_linearities.rectangular_to_polar_dict(
            y, residuals=True
        )
        y_hat = po.simul.non_linearities.polar_to_rectangular_dict(
            energy, state, residuals=True
        )
        for key in y:
            diff = y[key] - y_hat[key]
            assert torch.linalg.vector_norm(diff.flatten(), ord=2) < 1e-5

    def test_local_gain_control(self):
        x = torch.randn((10, 1, 256, 256), device=DEVICE)
        norm, direction = po.simul.non_linearities.local_gain_control(x)
        x_hat = po.simul.non_linearities.local_gain_release(norm, direction)
        diff = x - x_hat
        assert torch.linalg.vector_norm(diff.flatten(), ord=2) < 1e-4

    def test_local_gain_control_dict(self, basic_stim):
        spr = po.simul.SteerablePyramidFreq(
            basic_stim.shape[-2:],
            height=5,
            order=1,
            is_complex=False,
            tight_frame=True,
        ).to(DEVICE)
        y = spr(basic_stim)
        energy, state = po.simul.non_linearities.local_gain_control_dict(
            y, residuals=True
        )
        y_hat = po.simul.non_linearities.local_gain_release_dict(
            energy, state, residuals=True
        )
        for key in y:
            diff = y[key] - y_hat[key]
            assert torch.linalg.vector_norm(diff.flatten(), ord=2) < 1e-5


class TestLaplacianPyramid:
    def test_grad(self, basic_stim):
        lpyr = po.simul.LaplacianPyramid().to(DEVICE)
        y = lpyr.forward(basic_stim)
        assert y[0].requires_grad

    @pytest.mark.parametrize("n_scales", [3, 4, 5, 6])
    def test_synthesis(self, curie_img, n_scales):
        img = curie_img[
            :, :, 0:253, 0:234
        ]  # Original 256x256 shape is not good for testing padding
        lpyr = po.simul.LaplacianPyramid(n_scales=n_scales).to(DEVICE)
        y = lpyr.forward(img)
        img_recon = lpyr.recon_pyr(y)
        assert torch.allclose(img, img_recon)

    @pytest.mark.parametrize("n_scales", [3, 4, 5, 6])
    def test_match_pyrtools(self, curie_img, n_scales):
        img = curie_img[:, :, 0:253, 0:234]
        lpyr_po = po.simul.LaplacianPyramid(n_scales=n_scales).to(DEVICE)
        y_po = lpyr_po(img)
        lpyr_pt = pt.pyramids.LaplacianPyramid(img.squeeze().cpu(), height=n_scales)
        y_pt = [lpyr_pt.pyr_coeffs[(i, 0)] for i in range(n_scales)]
        assert len(y_po) == len(y_pt)
        for x_po, x_pt in zip(y_po, y_pt):
            x_po = x_po.squeeze().detach().cpu().numpy()
            assert np.abs(x_po - x_pt)[:-2, :-2].max() < 1e-5
            # The pyrtools implementation `pt.upConv performs`` padding after
            # upsampling. Our implementation `po.tools.upsample_convolve``
            # performs padding before upsampling, and, depending on the parity of
            # the image, sometimes performs additional zero padding after upsampling
            # up to one row/column. This causes inconsistency on the right and
            # bottom edges, so they are exluded in the comparison.


class TestFrontEnd:
    all_models = [
        "frontend.LinearNonlinear",
        "frontend.LuminanceGainControl",
        "frontend.LuminanceContrastGainControl",
        "frontend.OnOff",
    ]

    @pytest.mark.parametrize("model", all_models[:-1], indirect=True)
    def test_output_shape(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        assert model(img).shape == img.shape

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_gradient_flow(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE)
        y = model(img)
        assert y.requires_grad

    def test_onoff(self):
        po.simul.OnOff(7, pretrained=False).to(DEVICE)

    @pytest.mark.parametrize("kernel_size", [7, 31])
    @pytest.mark.parametrize("cache_filt", [False, True])
    def test_pretrained_onoff(self, kernel_size, cache_filt):
        if kernel_size != 31:
            with pytest.raises(ValueError):
                po.simul.OnOff(kernel_size, pretrained=True, cache_filt=cache_filt).to(
                    DEVICE
                )
        else:
            po.simul.OnOff(kernel_size, pretrained=True, cache_filt=cache_filt).to(
                DEVICE
            )

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_frontend_display_filters(self, model):
        fig = model.display_filters()
        plt.close(fig)

    @pytest.mark.parametrize("mdl", all_models)
    def test_kernel_size(self, mdl, einstein_img):
        kernel_size = 31
        if mdl == "frontend.LinearNonlinear":
            model = po.simul.LinearNonlinear(kernel_size, pretrained=True).to(DEVICE)
            model2 = po.simul.LinearNonlinear(
                (kernel_size, kernel_size), pretrained=True
            ).to(DEVICE)
        elif mdl == "frontend.LuminanceGainControl":
            model = po.simul.LuminanceGainControl(kernel_size, pretrained=True).to(
                DEVICE
            )
            model2 = po.simul.LuminanceGainControl(
                (kernel_size, kernel_size), pretrained=True
            ).to(DEVICE)
        elif mdl == "frontend.LuminanceContrastGainControl":
            model = po.simul.LuminanceContrastGainControl(
                kernel_size, pretrained=True
            ).to(DEVICE)
            model2 = po.simul.LuminanceContrastGainControl(
                (kernel_size, kernel_size), pretrained=True
            ).to(DEVICE)
        elif mdl == "frontend.OnOff":
            model = po.simul.OnOff(kernel_size, pretrained=True).to(DEVICE)
            model2 = po.simul.OnOff((kernel_size, kernel_size), pretrained=True).to(
                DEVICE
            )
        assert torch.allclose(model(einstein_img), model2(einstein_img)), (
            "Kernels somehow different!"
        )


class TestNaive:
    all_models = [
        "naive.Identity",
        "naive.Linear",
        "naive.Gaussian",
        "naive.CenterSurround",
    ]

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_gradient_flow(self, model):
        img = torch.ones(1, 1, 100, 100).to(DEVICE).requires_grad_()
        y = model(img)
        assert y.requires_grad

    @pytest.mark.parametrize(
        "mdl", ["naive.Linear", "naive.Gaussian", "naive.CenterSurround"]
    )
    def test_kernel_size(self, mdl, einstein_img):
        kernel_size = 10
        if mdl == "naive.Gaussian":
            model = po.simul.Gaussian(kernel_size, 1.0).to(DEVICE)
            model2 = po.simul.Gaussian((kernel_size, kernel_size), 1.0).to(DEVICE)
        elif mdl == "naive.Linear":
            model = po.simul.Linear(kernel_size).to(DEVICE)
            model2 = po.simul.Linear((kernel_size, kernel_size)).to(DEVICE)
        elif mdl == "naive.CenterSurround":
            model = po.simul.CenterSurround(kernel_size).to(DEVICE)
            model2 = po.simul.CenterSurround((kernel_size, kernel_size)).to(DEVICE)
        assert torch.allclose(model(einstein_img), model2(einstein_img)), (
            "Kernels somehow different!"
        )

    @pytest.mark.parametrize("mdl", ["naive.Gaussian", "naive.CenterSurround"])
    @pytest.mark.parametrize("cache_filt", [False, True])
    def test_cache_filt(self, cache_filt, mdl):
        img = torch.ones(1, 1, 100, 100).to(DEVICE).requires_grad_()
        if mdl == "naive.Gaussian":
            model = po.simul.Gaussian((31, 31), 1.0, cache_filt=cache_filt).to(DEVICE)
        elif mdl == "naive.CenterSurround":
            model = po.simul.CenterSurround((31, 31), cache_filt=cache_filt).to(DEVICE)
        model(img)  # forward pass should cache filt if True
        if cache_filt:
            assert model._filt is not None
        else:
            assert model._filt is None

    @pytest.mark.parametrize("center_std", [1.0, torch.as_tensor([1.0, 2.0])])
    @pytest.mark.parametrize("out_channels", [1, 2, 3])
    @pytest.mark.parametrize("on_center", [True, [True, False]])
    def test_CenterSurround_channels(self, center_std, out_channels, on_center):
        if not isinstance(center_std, float) and len(center_std) != out_channels:
            with pytest.raises(AssertionError):
                po.simul.CenterSurround(
                    (31, 31), center_std=center_std, out_channels=out_channels
                )
        else:
            po.simul.CenterSurround(
                (31, 31), center_std=center_std, out_channels=out_channels
            )

    def test_linear(self, basic_stim):
        model = po.simul.Linear().to(DEVICE)
        assert model(basic_stim).requires_grad


def convert_matlab_ps_rep_to_dict(
    vec: torch.Tensor,
    n_scales: int,
    n_orientations: int,
    spatial_corr_width: int,
    use_true_correlations: bool,
) -> OrderedDict:
    """Converts matlab vector of statistics to a dictionary.

    The matlab (and old plenoptic) PS representation includes a bunch of
    unnecessary and redundant stats, which are removed in our current
    implementation. This function converts that representation into a
    dictionary, which we then restrict to only those stats we now compute, so
    we can compare

    Parameters
    ----------
    vec
        1d vector of statistics.
    n_scales, n_orientations, spatial_corr_width, use_true_correlations
        Arguments used to initialize PS model

    Returns
    -------
    Dictionary of representation, with informative keys.

    """
    rep = OrderedDict()
    rep["pixel_statistics"] = OrderedDict()
    rep["pixel_statistics"] = vec[..., :6]

    n_filled = 6

    # magnitude_means
    rep["magnitude_means"] = OrderedDict()
    keys = (
        ["residual_highpass"]
        + [(sc, ori) for sc in range(n_scales) for ori in range(n_orientations)]
        + ["residual_lowpass"]
    )
    for ii, k in enumerate(keys):
        rep["magnitude_means"][k] = vec[..., n_filled + ii]
    n_filled += ii + 1

    # auto_correlation_magnitude
    nn = (
        spatial_corr_width,
        spatial_corr_width,
        n_scales,
        n_orientations,
    )
    # in the plenoptic version, auto_correlation_magnitude shape has n_scales and
    # n_orientations flipped relative to the matlab representation
    rep["auto_correlation_magnitude"] = (
        vec[..., n_filled : (n_filled + np.prod(nn))]
        .unflatten(-1, nn)
        .transpose(-1, -2)
    )
    n_filled += np.prod(nn)

    # skew_reconstructed & kurtosis_reconstructed
    nn = n_scales + 1
    rep["skew_reconstructed"] = vec[..., n_filled : (n_filled + nn)]
    n_filled += nn

    rep["kurtosis_reconstructed"] = vec[..., n_filled : (n_filled + nn)]
    n_filled += nn

    # auto_correlation_reconstructed
    nn = (spatial_corr_width, spatial_corr_width, (n_scales + 1))
    rep["auto_correlation_reconstructed"] = vec[
        ..., n_filled : (n_filled + np.prod(nn))
    ].unflatten(-1, nn)
    n_filled += np.prod(nn)

    if use_true_correlations:
        nn = n_scales + 1
        rep["std_reconstructed"] = vec[..., n_filled : (n_filled + nn)]
        n_filled += nn
    else:
        # place a dummy entry, so the order of keys is correct
        rep["std_reconstructed"] = []

    # cross_orientation_correlation_magnitude
    nn = (n_orientations, n_orientations, (n_scales + 1))
    rep["cross_orientation_correlation_magnitude"] = vec[
        ..., n_filled : (n_filled + np.prod(nn))
    ].unflatten(-1, nn)
    n_filled += np.prod(nn)

    if use_true_correlations:
        nn = (n_orientations, n_scales)
        rep["magnitude_std"] = vec[..., n_filled : (n_filled + np.prod(nn))].unflatten(
            -1, nn
        )
        n_filled += np.prod(nn)
    else:
        # place a dummy entry, so the order of keys is correct
        rep["magnitude_std"] = []

    # cross_scale_correlation_magnitude
    nn = (n_orientations, n_orientations, n_scales)
    rep["cross_scale_correlation_magnitude"] = vec[
        ..., n_filled : (n_filled + np.prod(nn))
    ].unflatten(-1, nn)
    n_filled += np.prod(nn)

    # cross_orientation_correlation_real
    nn = (
        max(2 * n_orientations, 5),
        max(2 * n_orientations, 5),
        (n_scales + 1),
    )
    rep["cross_orientation_correlation_real"] = vec[
        ..., n_filled : (n_filled + np.prod(nn))
    ].unflatten(-1, nn)
    n_filled += np.prod(nn)

    # cross_scale_correlation_real
    nn = (2 * n_orientations, max(2 * n_orientations, 5), n_scales)
    rep["cross_scale_correlation_real"] = vec[
        ..., n_filled : (n_filled + np.prod(nn))
    ].unflatten(-1, nn)
    n_filled += np.prod(nn)

    # var_highpass_residual
    rep["var_highpass_residual"] = vec[..., n_filled]

    return rep


def construct_normalizing_dict(
    plen_ps: po.simul.PortillaSimoncelli, img: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Construct dictionary to normalize covariances in PS representation.

    The matlab code computes covariances instead of correlations for the
    cross-orientation and cross-scale correlations. Here, we construct the
    tensors required to normalize those covariances to correlations, which are
    the outer product of the variances of the tensors that created them (which
    are intermediaries and not present in the final PS model output)

    """
    coeffs = plen_ps._compute_pyr_coeffs(img)[1]
    mags, reals = plen_ps._compute_intermediate_representations(coeffs)
    doub_mags, doub_sep = plen_ps._double_phase_pyr_coeffs(coeffs)
    mags_var = torch.stack([m.var((-2, -1), correction=0) for m in mags], -1)

    normalizing_dict = {}
    com = einops.einsum(mags_var, mags_var, "b c o1 s, b c o2 s -> b c o1 o2 s")
    normalizing_dict["cross_orientation_correlation_magnitude"] = com.pow(0.5)

    if plen_ps.n_scales > 1:
        doub_mags_var = torch.stack(
            [m.var((-2, -1), correction=0) for m in doub_mags], -1
        )
        reals_var = torch.stack([r.var((-2, -1), correction=0) for r in reals], -1)
        doub_sep_var = torch.stack(
            [s.var((-2, -1), correction=0) for s in doub_sep], -1
        )
        csm = einops.einsum(
            mags_var[..., :-1],
            doub_mags_var,
            "b c o1 s, b c o2 s -> b c o1 o2 s",
        )
        normalizing_dict["cross_scale_correlation_magnitude"] = csm.pow(0.5)
        csr = einops.einsum(
            reals_var[..., :-1],
            doub_sep_var,
            "b c o1 s, b c o2 s -> b c o1 o2 s",
        )
        normalizing_dict["cross_scale_correlation_real"] = csr.pow(0.5)
    else:
        normalizing_dict["cross_scale_correlation_magnitude"] = 1
        normalizing_dict["cross_scale_correlation_real"] = 1

    return normalizing_dict


def remove_redundant_and_normalize(
    matlab_rep: OrderedDict,
    use_true_correlations: bool,
    plen_ps: po.simul.PortillaSimoncelli,
    normalizing_dict: dict,
) -> torch.Tensor:
    """Remove redundant stats from dictionary of representation, and normalize
    correlations

    Redundant stats fall in two categories: those that are not included at all
    anymore (e.g., magnitude means, extra zero placeholders), and those that
    are computed automatically (because of how the computation works) and then
    discarded (e.g., symmetries in autocorrelation). This function removes both.

    Additionally, if use_true_correlations=False, we normalize the
    correlations. The matlab PS code did not do so, so that, e.g., the center
    value of the autocorrelations was the corresponding variance (rather than
    being normalized to 1). Originally, we supported both normalized and
    un-normalized, but now we only support the normalized version. Thus, to
    compare with the matlab outputs (and older plenoptic versions), need to
    normalize the auto- and cross-correlations.

    We also grab the center values of auto_correlation_reconstructed and
    create the new statistic std_reconstructed, as this information is
    important.

    Finally, we take the negative of half the cross_scale_correlation_real.
    When doubling the phase, Portilla-Simoncelli accidentally calculated the
    negative of the real component: they called atan2(real, imag), whereas
    atan2 receives the y-value (thus, the imaginary component) first. This
    caused the correlations between the real coefficients on one scale and the
    real coefficients on the next to be negative where they should be positive,
    and vice versa (this doesn't affect the magnitudes or the imaginary
    components on the next scale).

    """
    # Remove those stats that are not included at all.
    matlab_rep.pop("magnitude_means")
    matlab_rep.pop("cross_orientation_correlation_real")

    # Remove the 0 placeholders
    matlab_rep["cross_scale_correlation_magnitude"] = matlab_rep[
        "cross_scale_correlation_magnitude"
    ][..., :-1]
    matlab_rep["cross_orientation_correlation_magnitude"] = matlab_rep[
        "cross_orientation_correlation_magnitude"
    ][..., :-1]
    matlab_rep["cross_scale_correlation_real"] = matlab_rep[
        "cross_scale_correlation_real"
    ][..., : plen_ps.n_orientations, :, :-1]
    # if there are two orientations, there's some more 0 placeholders
    if plen_ps.n_orientations == 2:
        matlab_rep["cross_scale_correlation_real"] = matlab_rep[
            "cross_scale_correlation_real"
        ][..., :-1, :]
    # See docstring for why we make these specific stats negative
    matlab_rep["cross_scale_correlation_real"][
        ..., : plen_ps.n_orientations, :
    ] = -matlab_rep["cross_scale_correlation_real"][..., : plen_ps.n_orientations, :]

    if not use_true_correlations:
        # Create std_reconstructed
        ctr_ind = plen_ps.spatial_corr_width // 2
        var_recon = matlab_rep["auto_correlation_reconstructed"][
            ..., ctr_ind, ctr_ind, :
        ].clone()
        matlab_rep["std_reconstructed"] = var_recon**0.5

        # Normalize the autocorrelations using their center values
        matlab_rep["auto_correlation_reconstructed"] /= var_recon
        acm_ctr = matlab_rep["auto_correlation_magnitude"][
            ..., ctr_ind, ctr_ind, :, :
        ].clone()
        matlab_rep["auto_correlation_magnitude"] /= acm_ctr

        # Create magnitude_std
        diag = torch.arange(plen_ps.n_orientations)
        var_mags = matlab_rep["cross_orientation_correlation_magnitude"][
            ..., diag, diag, :
        ]
        matlab_rep["magnitude_std"] = var_mags.pow(0.5)

        # The cross-correlations are normalized by the outer product of the
        # variances of the tensors that create them. We have created these and
        # saved them in normalizing dict, which we use here
        crosscorr_keys = [
            "cross_scale_correlation_real",
            "cross_scale_correlation_magnitude",
            "cross_orientation_correlation_magnitude",
        ]
        for k in crosscorr_keys:
            matlab_rep[k] = matlab_rep[k] / normalizing_dict[k]

    # Finally, turn dict back into vector, removing redundant stats
    return plen_ps.convert_to_tensor(matlab_rep)


class TestPortillaSimoncelli:
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    def test_portilla_simoncelli(
        self,
        n_scales,
        n_orientations,
        spatial_corr_width,
        einstein_img,
    ):
        ps = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        ps(einstein_img)

    # tests for whether output matches the original matlab output.  This implicitly
    # tests that Portilla_simoncelli.forward() returns an object of the correct size.
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", [3, 5, 7, 9])
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_ps_torch_v_matlab(
        self,
        n_scales,
        n_orientations,
        spatial_corr_width,
        im,
        portilla_simoncelli_matlab_test_vectors,
    ):
        # the matlab outputs were computed on images with values between 0 and
        # 255 (not 0 and 1, which is what po.load_images does by default). Note
        # that for the einstein-9-2-4, einstein-9-3-4, einstein-9-4-4,
        # multiplying by 255 before converting to float64 (rather than
        # converting to float64 and then multiplying by 255) matters, because
        # floating points are fun.
        im0 = 255 * po.load_images(IMG_DIR / "256" / f"{im}.pgm")
        im0 = im0.to(torch.float64).to(DEVICE)
        ps = (
            po.simul.PortillaSimoncelli(
                im0.shape[-2:],
                n_scales=n_scales,
                n_orientations=n_orientations,
                spatial_corr_width=spatial_corr_width,
            )
            .to(DEVICE)
            .to(torch.float64)
        )
        python_vector = ps(im0)

        matlab_rep = sio.loadmat(
            f"{portilla_simoncelli_matlab_test_vectors}/"
            f"{im}-scales{n_scales}-ori{n_orientations}"
            f"-spat{spatial_corr_width}.mat"
        )
        matlab_rep = (
            torch.from_numpy(matlab_rep["params_vector"].flatten())
            .unsqueeze(0)
            .unsqueeze(0)
        )
        matlab_rep = convert_matlab_ps_rep_to_dict(
            matlab_rep.to(DEVICE),
            n_scales,
            n_orientations,
            spatial_corr_width,
            False,
        )
        norm_dict = construct_normalizing_dict(ps, im0)
        matlab_rep = remove_redundant_and_normalize(matlab_rep, False, ps, norm_dict)
        matlab_rep = po.to_numpy(matlab_rep).squeeze()
        python_vector = po.to_numpy(python_vector).squeeze()

        np.testing.assert_allclose(python_vector, matlab_rep, rtol=1e-4, atol=1e-4)

    # tests for whether output matches the saved python output. This implicitly
    # tests that Portilla_simoncelli.forward() returns an object of the correct
    # size.
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_ps_torch_output(
        self,
        n_scales,
        n_orientations,
        spatial_corr_width,
        im,
        portilla_simoncelli_test_vectors,
    ):
        im0 = po.load_images(IMG_DIR / "256" / f"{im}.pgm")
        im0 = im0.to(torch.float64).to(DEVICE)
        ps = (
            po.simul.PortillaSimoncelli(
                im0.shape[-2:],
                n_scales=n_scales,
                n_orientations=n_orientations,
                spatial_corr_width=spatial_corr_width,
            )
            .to(DEVICE)
            .to(torch.float64)
        )
        output = ps(im0)

        saved = np.load(
            f"{portilla_simoncelli_test_vectors}/"
            f"{im}_scales-{n_scales}_ori-{n_orientations}_"
            f"spat-{spatial_corr_width}.npy"
        )

        output = po.to_numpy(output)
        np.testing.assert_allclose(output, saved, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    def test_ps_convert(
        self, n_scales, n_orientations, spatial_corr_width, einstein_img
    ):
        ps = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        rep = ps(einstein_img)
        assert torch.all(rep == ps.convert_to_tensor(ps.convert_to_dict(rep))), (
            "Convert to tensor or dict is broken!"
        )

    def test_ps_synthesis(self, portilla_simoncelli_synthesize, run_test=True):
        """Test PS texture metamer synthesis.

        Parameters
        ----------
        portilla_simoncelli_synthesize : str
            Path to the .npz file to test against
        run_test : bool, optional
            If True, we run the test, comparing the current synthesis against
            the saved results. If False, we don't run the test, and return the
            Metamer object instead (used when updating the file to test
            against, as in tests/utils.update_ps_synthesis_test_file)

        """
        # this tests whether the output of metamer synthesis is consistent.
        # this is probably the most likely to fail as our requirements change,
        # because if something in how torch computes its gradients changes,
        # then our outputs will change. for example, in release 1.10
        # (https://github.com/pytorch/pytorch/releases/tag/v1.10.0), they fixed
        # the sub-gradient for torch.a{max,min}, which resulted in our PS
        # synthesis getting worse, somehow. this is just a note to keep an eye
        # on this; you might need to update the output to test against as
        # versions change. you probably only need to store the most recent
        # version, because that's what we test against.
        torch.use_deterministic_algorithms(True)
        with np.load(portilla_simoncelli_synthesize) as f:
            im = f["im"]
            im_init = f["im_init"]
            im_synth = f["im_synth"]
            rep_synth = f["rep_synth"]

        im0 = torch.as_tensor(im).unsqueeze(0).unsqueeze(0).to(DEVICE).to(torch.float64)
        model = (
            po.simul.PortillaSimoncelli(
                im0.shape[-2:],
                n_scales=4,
                n_orientations=4,
                spatial_corr_width=9,
            )
            .to(DEVICE)
            .to(torch.float64)
        )

        po.tools.set_seed(1)
        im_init = torch.as_tensor(im_init).unsqueeze(0).unsqueeze(0)
        met = po.synth.MetamerCTF(
            im0,
            model,
            initial_image=im_init,
            loss_function=po.tools.optim.l2_norm,
            range_penalty_lambda=0,
            coarse_to_fine="together",
        )

        # this is the same as the default optimizer, but we explicitly
        # instantiate it anyway, in case we change the defaults at some point
        optim = torch.optim.Adam([met.metamer], lr=0.01, amsgrad=True)
        met.synthesize(
            max_iter=200,
            optimizer=optim,
            change_scale_criterion=None,
            ctf_iters_to_check=15,
        )

        output = met.metamer
        if run_test:
            np.testing.assert_allclose(
                po.to_numpy(output).squeeze(),
                im_synth.squeeze(),
                rtol=1e-4,
                atol=1e-4,
            )

            np.testing.assert_allclose(
                po.to_numpy(model(output)).squeeze(),
                rep_synth.squeeze(),
                rtol=1e-4,
                atol=1e-4,
            )
        else:
            return met

    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    def test_portilla_simoncelli_scales(
        self,
        n_scales,
        n_orientations,
        spatial_corr_width,
        portilla_simoncelli_scales,
    ):
        with np.load(portilla_simoncelli_scales, allow_pickle=True) as f:
            key = f"scale-{n_scales}_ori-{n_orientations}_width-{spatial_corr_width}"
            saved = f[key]

        model = po.simul.PortillaSimoncelli(
            [256, 256],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)

        output = model._representation_scales

        np.testing.assert_equal(output, saved)

    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("img_size", [255, 254, 252, 160])
    def test_other_size_images(self, n_scales, img_size):
        im0 = po.load_images(IMG_DIR / "256" / "nuts.pgm").to(DEVICE)
        im0 = im0[..., :img_size, :img_size]
        if any([(img_size / 2**i) % 2 for i in range(n_scales)]):
            expectation = pytest.raises(
                ValueError,
                match=(
                    "Because of how the Portilla-Simoncelli model handles multiscale"
                ),
            )
        else:
            expectation = does_not_raise()
        with expectation:
            model = po.simul.PortillaSimoncelli(
                im0.shape[-2:],
                n_scales=n_scales,
            ).to(DEVICE)
            model(im0)

    @pytest.mark.parametrize("img_size", [160, 128])
    def test_nonsquare_images(self, img_size):
        im0 = po.load_images(IMG_DIR / "256" / "nuts.pgm").to(DEVICE)
        im0 = im0[..., :img_size]
        model = po.simul.PortillaSimoncelli(
            im0.shape[-2:],
            # with height 4, spatial_corr_width=9 is too big for final scale
            # and image size 128
            spatial_corr_width=7,
        ).to(DEVICE)
        model(im0)

    @pytest.mark.parametrize("batch_channel", [(1, 3), (2, 1), (2, 3)])
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    def test_multibatchchannel(
        self,
        batch_channel,
        n_scales,
        n_orientations,
        spatial_corr_width,
        einstein_img,
    ):
        model = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        rep = model(einstein_img.repeat((*batch_channel, 1, 1)))
        if rep.shape[:2] != batch_channel:
            raise ValueError(
                "Output doesn't have same number of batch/channel dims as input!"
            )

    @pytest.mark.parametrize("batch_channel", [(1, 1), (1, 3), (2, 1), (2, 3)])
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    def test_plot_representation(
        self,
        batch_channel,
        n_scales,
        n_orientations,
        spatial_corr_width,
        einstein_img,
    ):
        model = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        model.plot_representation(
            model(einstein_img.repeat((*batch_channel, 1, 1))),
            title="Representation",
        )

    def test_update_plot(self, einstein_img):
        model = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
        ).to(DEVICE)
        _, axes = model.plot_representation(model(einstein_img))
        orig_y = axes[0].containers[0].markerline.get_ydata()
        img = po.load_images(IMG_DIR / "256" / "nuts.pgm").to(DEVICE)
        artists = model.update_plot(axes, model(img).cpu())
        updated_y = artists[0].get_ydata()
        if np.equal(orig_y, updated_y).all():
            raise ValueError("Update plot didn't run successfully!")

    @pytest.mark.parametrize("batch_channel", [(1, 1), (1, 3), (2, 1), (2, 3)])
    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    def test_plot_representation_dim_assumption(
        self,
        batch_channel,
        n_scales,
        n_orientations,
        spatial_corr_width,
        einstein_img,
    ):
        # there's an assumption I make in plot_representation that I want to
        # ensure is tested
        model = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        rep = model(einstein_img.repeat((*batch_channel, 1, 1)))
        rep = model.convert_to_dict(rep[0].unsqueeze(0).mean(1, keepdim=True))
        if any([v.ndim < 3 for v in rep.values()]):
            raise ValueError("Somehow this doesn't have at least 3 dimensions!")
        if any([v.shape[:2] != (1, 1) for v in rep.values()]):
            raise ValueError("Somehow this has an extra batch or channel!")

    # fft doesn't support float16, so we can't support it
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes(self, dtype, einstein_img):
        model = po.simul.PortillaSimoncelli(einstein_img.shape[-2:]).to(DEVICE)
        model(einstein_img.to(dtype))

    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    def test_scales_shapes(
        self, n_scales, n_orientations, spatial_corr_width, einstein_img
    ):
        # test that the shapes we use to assign scale labels to each statistic
        # and determine redundant stats are accurate
        model = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        # this hack is to prevent model from removing redundant stats
        model._necessary_stats_mask = None
        rep = model(einstein_img)
        # and then we get them back into their original shapes
        unpacked_rep = einops.unpack(rep, model._pack_info, "b c *")
        # because _necessary_stats_dict is an ordered dictionary, its elements
        # will be in the same order as in unpackaged_rep
        for unp_v, dict_v in zip(unpacked_rep, model._necessary_stats_dict.values()):
            # when we have a single scale, _necessary_stats_dict will contain
            # keys for the cross_scale correlations, but there are no
            # corresponding values. Thus, skip.
            if dict_v.nelement() == 0:
                continue
            # ignore batch and channel
            unp_v = unp_v[0, 0]
            if not unp_v.shape:
                # then this is var_residual_highpass, which has a single element
                np.testing.assert_equal(unp_v.nelement(), dict_v.nelement())
            else:
                np.testing.assert_equal(unp_v.shape, dict_v.shape)

    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_redundancies(self, n_scales, n_orientations, spatial_corr_width, im):
        # test that the computed statistics have the redundancies we think they
        # do
        im = po.load_images(IMG_DIR / "256" / f"{im}.pgm")
        im = im.to(torch.float64).to(DEVICE)
        model = po.simul.PortillaSimoncelli(
            im.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        # this hack is to prevent model from removing redundant stats
        model._necessary_stats_mask = None
        rep = model(im)
        # and then we get them back into their original shapes (with lots of
        # redundancies)
        unpacked_rep = einops.unpack(rep, model._pack_info, "b c *")
        for unp_v, (k, nec_v) in zip(unpacked_rep, model._necessary_stats_dict.items()):
            # find the redundant values for this stat
            red_v = torch.logical_not(nec_v)
            # then there are no redundant values here
            if red_v.sum() == 0:
                continue
            unp_vals = []
            mask_vals = []
            ctr_vals = []
            for sc in range(red_v.shape[-1]):
                red_idx = torch.stack(torch.where(red_v[..., sc]), -1)
                if red_idx.shape[-1] == 3:
                    # auto_correlation_magnitude has an extra dimension
                    # compared to the others ignore batch and channel
                    assert k == "auto_correlation_magnitude", (
                        f"Somehow got extra dimension for {k}!"
                    )
                    # then drop the duplicates
                    red_idx = torch.unique(red_idx[..., :2], dim=0)
                val = unp_v[0, 0, ..., sc]
                if k == "cross_orientation_correlation_magnitude":
                    # Symmetry M_{i,j} = M_{j,i}.
                    for i in red_idx:
                        unp_vals.append(val[i[0], i[1]])
                        mask_vals.append(val[i[1], i[0]])
                elif k.startswith("auto_correlation"):
                    # center values of autocorrelations should be 1
                    ctr_vals.append(
                        val[
                            model.spatial_corr_width // 2,
                            model.spatial_corr_width // 2,
                        ]
                    )
                    # Symmetry M_{i,j} = M_{n-i+1, n-j+1}
                    for i in red_idx:
                        unp_vals.append(val[i[0], i[1]])
                        # need to change where we index into depending on
                        # whether spatial_corr_width (and thus the shape of
                        # val) is even or odd
                        offset = 0 if not spatial_corr_width % 2 else 1

                        mask_vals.append(val[-(i[0] + offset), -(i[1] + offset)])
                else:
                    raise ValueError(f"stat {k} unexpectedly has redundant values!")
            # and check for equality
            if ctr_vals:
                ctr_vals = torch.stack(ctr_vals)
                torch.equal(ctr_vals, torch.ones_like(ctr_vals))
            unp_vals = torch.stack(unp_vals)
            mask_vals = torch.stack(mask_vals)
            torch.testing.assert_close(unp_vals, mask_vals, atol=1e-6, rtol=1e-7)

    @pytest.mark.parametrize("n_scales", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_orientations", [2, 3, 4])
    @pytest.mark.parametrize("spatial_corr_width", range(3, 10))
    @pytest.mark.parametrize("im", ["curie", "einstein", "metal", "nuts"])
    def test_crosscorrs(self, n_scales, n_orientations, spatial_corr_width, im):
        # test that cross-correlations we compute are actual cross correlations
        im = po.load_images(IMG_DIR / "256" / f"{im}.pgm")
        im = im.to(torch.float64).to(DEVICE)
        model = po.simul.PortillaSimoncelli(
            im.shape[-2:],
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        ).to(DEVICE)
        # this hack is to prevent model from removing redundant stats, which
        # insert NaNs, making the comparison difficult
        model._necessary_stats_mask = None
        rep = model(im)
        # and then we get them back into their original shapes (with lots of
        # redundancies)
        unpacked_rep = einops.unpack(rep, model._pack_info, "b c *")
        keys = list(model._necessary_stats_dict.keys())
        # need to get the intermediates necessary for testing
        # cross-correlations
        coeffs = model._compute_pyr_coeffs(im)[1]
        mags, reals = model._compute_intermediate_representations(coeffs)
        doub_mags, doub_sep = model._double_phase_pyr_coeffs(coeffs)
        # the cross-orientation correlations
        torch_corrs = []
        for m in mags:
            m = einops.rearrange(m, "b c o h w -> (b c o) (h w)")
            torch_corrs.append(torch.corrcoef(m).unsqueeze(0).unsqueeze(0))
        torch_corr = torch.stack(torch_corrs, -1)
        idx = keys.index("cross_orientation_correlation_magnitude")
        torch.testing.assert_close(unpacked_rep[idx], torch_corr, atol=0, rtol=1e-12)
        # only have cross-scale correlations when there's more than one scale
        if n_scales > 1:
            # cross-scale magnitude correlations
            torch_corrs = []
            for m, d in zip(mags[:-1], doub_mags):
                concat = torch.cat([m, d], dim=2)
                concat = einops.rearrange(concat, "b c o h w -> (b c o) (h w)")
                # this matrix contains the 4 sub-matrices, each of shape
                # (n_orientations, n_orientations), only one of which we want:
                # the correlations between the magnitudes at this scale and the
                # doubled ones at the next scale.
                c = torch.corrcoef(concat)[:n_orientations, n_orientations:]
                torch_corrs.append(c.unsqueeze(0).unsqueeze(0))
            torch_corr = torch.stack(torch_corrs, -1)
            idx = keys.index("cross_scale_correlation_magnitude")
            torch.testing.assert_close(
                unpacked_rep[idx], torch_corr, atol=0, rtol=1e-12
            )
            # cross-scale real correlations
            torch_corrs = []
            for r, s in zip(reals[:-1], doub_sep):
                concat = torch.cat([r, s], dim=2)
                concat = einops.rearrange(concat, "b c o h w -> (b c o) (h w)")
                # this matrix contains the 4 sub-matrices, only one of which we
                # want: the correlations between the real coeffs at this scale
                # and the doubled real and imaginary coeffs at the next scale.
                # the reals have n_orientations orientations, while the
                # doub_sep have twice that (because they contain both the real
                # and imaginary)
                c = torch.corrcoef(concat)[:n_orientations, n_orientations:]
                torch_corrs.append(c.unsqueeze(0).unsqueeze(0))
            torch_corr = torch.stack(torch_corrs, -1)
            idx = keys.index("cross_scale_correlation_real")
            torch.testing.assert_close(
                unpacked_rep[idx], torch_corr, atol=1e-5, rtol=2e-5
            )

    def test_convert_to_dict_error_diff_model(self, einstein_img):
        ps = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=4,
        ).to(DEVICE)
        rep = ps(einstein_img)
        ps = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
            n_scales=2,
        ).to(DEVICE)
        with pytest.raises(
            ValueError, match="representation tensor is the wrong length"
        ):
            ps.convert_to_dict(rep)

    def test_convert_to_dict_error(self, einstein_img):
        ps = po.simul.PortillaSimoncelli(
            einstein_img.shape[-2:],
        ).to(DEVICE)
        rep = ps(einstein_img)
        with pytest.raises(
            ValueError, match="representation tensor is the wrong length"
        ):
            ps.convert_to_dict(rep[..., :-10])


class TestFilters:
    @pytest.mark.parametrize(
        "std", [5.0, torch.as_tensor(1.0, device=DEVICE), -1.0, 0.0]
    )
    @pytest.mark.parametrize("kernel_size", [(31, 31), (3, 2), (7, 7), 5])
    @pytest.mark.parametrize("out_channels", [1, 3, 10])
    def test_circular_gaussian2d_shape(self, std, kernel_size, out_channels):
        if std <= 0.0:
            with pytest.raises(AssertionError):
                circular_gaussian2d((7, 7), std)
        else:
            filt = circular_gaussian2d(kernel_size, std, out_channels)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            assert filt.shape == (out_channels, 1, *kernel_size)
            assert filt.sum().isclose(torch.ones(1, device=DEVICE) * out_channels)

    def test_circular_gaussian2d_wrong_std_length(self):
        std = torch.as_tensor([1.0, 2.0], device=DEVICE)
        out_channels = 3
        with pytest.raises(AssertionError):
            circular_gaussian2d((7, 7), std, out_channels)

    @pytest.mark.parametrize("kernel_size", [5, 11, 20])
    @pytest.mark.parametrize(
        "std,expectation",
        [
            (1.0, does_not_raise()),
            (20.0, does_not_raise()),
            (0.0, pytest.raises(ValueError, match="must be positive")),
            (1, does_not_raise()),
            ([1, 1], pytest.raises(ValueError, match="must have only one element")),
            (torch.tensor(1), does_not_raise()),
            (torch.tensor([1]), does_not_raise()),
            (
                torch.tensor([1, 1]),
                pytest.raises(ValueError, match="must have only one element"),
            ),
        ],
    )
    def test_gaussian1d(self, kernel_size, std, expectation):
        with expectation:
            filt = gaussian1d(kernel_size, std)
            assert filt.sum().isclose(torch.ones(1))
            assert filt.shape == torch.Size([kernel_size])
