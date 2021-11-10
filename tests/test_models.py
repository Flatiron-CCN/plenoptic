#!/usr/bin/env python3
from math import pi
import matplotlib.pyplot as plt
import plenoptic
import plenoptic as po
import pytest
import torch

import plenoptic
from plenoptic.simulate.canonical_computations import (gaussian1d, circular_gaussian2d)
from conftest import DEVICE, DATA_DIR


@pytest.fixture()
def image_input():
    return torch.rand(1, 1, 100, 100)


class TestNonLinearities(object):
    def test_rectangular_to_polar_dict(self, basic_stim):
        spc = po.simul.Steerable_Pyramid_Freq(basic_stim.shape[-2:], height=5,
                                              order=1, is_complex=True, tight_frame=True).to(DEVICE)
        y = spc(basic_stim)
        energy, state = po.simul.non_linearities.rectangular_to_polar_dict(y, residuals=True)
        y_hat = po.simul.non_linearities.polar_to_rectangular_dict(energy, state, residuals=True)
        for key in y.keys():
            assert torch.norm(y[key] - y_hat[key]) < 1e-5

    def test_local_gain_control(self):
        x = torch.randn((10, 1, 256, 256), device=DEVICE)
        norm, direction = po.simul.non_linearities.local_gain_control(x)
        x_hat = po.simul.non_linearities.local_gain_release(norm, direction)
        assert torch.norm(x - x_hat) < 1e-4

    def test_local_gain_control_dict(self, basic_stim):
        spr = po.simul.Steerable_Pyramid_Freq(basic_stim.shape[-2:], height=5,
                                              order=1, is_complex=False, tight_frame=True).to(DEVICE)
        y = spr(basic_stim)
        energy, state = po.simul.non_linearities.local_gain_control_dict(y, residuals=True)
        y_hat = po.simul.non_linearities.local_gain_release_dict(energy, state, residuals=True)
        for key in y.keys():
            assert torch.norm(y[key] - y_hat[key]) < 1e-5


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
        mdl = po.simul.OnOff(7, pretrained=False).to(DEVICE)

    @pytest.mark.parametrize("kernel_size", [7, 31])
    @pytest.mark.parametrize("cache_filt", [False, True])
    def test_pretrained_onoff(self, kernel_size, cache_filt):
        if kernel_size != 31:
            with pytest.raises(AssertionError):
                mdl = po.simul.OnOff(kernel_size, pretrained=True, cache_filt=cache_filt).to(DEVICE)
        else:
            mdl = po.simul.OnOff(kernel_size, pretrained=True, cache_filt=cache_filt).to(DEVICE)

    @pytest.mark.parametrize("model", all_models, indirect=True)
    def test_frontend_display_filters(self, model):
        fig = model.display_filters()
        plt.close(fig)


class TestNaive(object):

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

    @pytest.mark.parametrize("mdl", ["naive.Gaussian", "naive.CenterSurround"])
    @pytest.mark.parametrize("cache_filt", [False, True])
    def test_cache_filt(self, cache_filt, mdl):
        img = torch.ones(1, 1, 100, 100).to(DEVICE).requires_grad_()
        if mdl == "naive.Gaussian":
            model = po.simul.Gaussian((31, 31), 1., cache_filt=cache_filt).to(DEVICE)
        elif mdl == "naive.CenterSurround":
            model = po.simul.CenterSurround((31, 31), cache_filt=cache_filt).to(DEVICE)

        y = model(img)  # forward pass should cache filt if True

        if cache_filt:
            assert model._filt is not None
        else:
            assert model._filt is None

    @pytest.mark.parametrize("center_std", [1., torch.tensor([1., 2.])])
    @pytest.mark.parametrize("out_channels", [1, 2, 3])
    @pytest.mark.parametrize("on_center", [True, [True, False]])
    def test_CenterSurround_channels(self, center_std, out_channels, on_center):
        if not isinstance(center_std, float) and len(center_std) != out_channels:
            with pytest.raises(AssertionError):
                model = po.simul.CenterSurround((31, 31), center_std=center_std, out_channels=out_channels)
        else:
            model = po.simul.CenterSurround((31, 31), center_std=center_std, out_channels=out_channels)

    def test_linear(self, basic_stim):
        model = plenoptic.simul.Linear().to(DEVICE)
        assert model(basic_stim).requires_grad

    def test_linear_metamer(self, einstein_img):
        model = plenoptic.simul.Linear().to(DEVICE)
        M = po.synth.Metamer(einstein_img, model)
        M.synthesize(max_iter=3)


class TestLaplacianPyramid(object):

    def test_grad(self, basic_stim):
        L = po.simul.Laplacian_Pyramid().to(DEVICE)
        y = L.analysis(basic_stim)
        assert y[0].requires_grad


class TestFilters:
    @pytest.mark.parametrize("std", [5., torch.tensor(1.), -1., 0.])
    @pytest.mark.parametrize("kernel_size", [(31, 31), (3, 2), (7, 7), 5])
    @pytest.mark.parametrize("out_channels", [1, 3, 10])
    def test_circular_gaussian2d_shape(self, std, kernel_size, out_channels):

        if std <=0.:
            with pytest.raises(AssertionError):
                circular_gaussian2d((7, 7), std)
        else:
            filt = circular_gaussian2d(kernel_size, std, out_channels)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            assert filt.shape == (out_channels, 1, *kernel_size)
            assert filt.sum().isclose(torch.ones(1) * out_channels)

    def test_circular_gaussian2d_wrong_std_length(self):
        std = torch.tensor([1., 2.])
        out_channels = 3
        with pytest.raises(AssertionError):
            circular_gaussian2d((7, 7), std, out_channels)


    @pytest.mark.parametrize("kernel_size", [5, 11, 20])
    @pytest.mark.parametrize("std", [1., 20., 0.])
    def test_gaussian1d(self, kernel_size, std):
        if std <=0:
            with pytest.raises(AssertionError):
                gaussian1d(kernel_size, std)
        else:
            filt = gaussian1d(kernel_size, std)
            assert filt.sum().isclose(torch.ones(1))
            assert filt.shape == torch.Size([kernel_size])
