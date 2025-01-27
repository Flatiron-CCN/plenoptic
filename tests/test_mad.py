# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import os.path as op

import matplotlib as mpl
import numpy as np
import pytest
import torch

import plenoptic as po
from conftest import DEVICE

# use the html backend, so we don't need to have ffmpeg
mpl.rcParams["animation.writer"] = "html"
mpl.use("agg")


# in order for pickling to work with functions, they must be defined at top of
# module: https://stackoverflow.com/a/36995008
def rgb_mse(*args):
    return po.metric.mse(*args).mean()


def rgb_l2_norm(*args):
    return po.tools.optim.l2_norm(*args).mean()


# MAD requires metrics are *dis*-similarity metrics, so that they
# return 0 if two images are identical (SSIM normally returns 1)
def dis_ssim(*args):
    return (1 - po.metric.ssim(*args)).mean()


class ModuleMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mdl = po.metric.NLP()

    def forward(self, x, y):
        return (self.mdl(x) - self.mdl(y)).abs().mean()


class NonModuleMetric:
    def __init__(self):
        self.name = "nonmodule"

    def __call__(self, x, y):
        return (x - y).abs().sum()


class TestMAD:
    @pytest.mark.parametrize("target", ["min", "max"])
    @pytest.mark.parametrize("model_order", ["mse-ssim", "ssim-mse"])
    @pytest.mark.parametrize("store_progress", [False, True, 2])
    def test_basic(self, curie_img, target, model_order, store_progress):
        if model_order == "mse-ssim":
            model = po.metric.mse
            model2 = dis_ssim
        elif model_order == "ssim-mse":
            model = dis_ssim
            model2 = po.metric.mse
        mad = po.synth.MADCompetition(curie_img, model, model2, target)
        mad.synthesize(max_iter=5, store_progress=store_progress)
        if store_progress:
            mad.synthesize(max_iter=5, store_progress=store_progress)

    @pytest.mark.parametrize(
        "fail", [False, "img", "metric1", "metric2", "target", "tradeoff"]
    )
    @pytest.mark.parametrize("rgb", [False, True])
    def test_save_load(self, curie_img, fail, rgb, tmp_path):
        # this works with either rgb or grayscale images
        metric = rgb_mse
        if rgb:
            curie_img = curie_img.repeat(1, 3, 1, 1)
            metric2 = rgb_l2_norm
        else:
            metric2 = dis_ssim
        target = "min"
        tradeoff = 1
        mad = po.synth.MADCompetition(
            curie_img, metric, metric2, target, metric_tradeoff_lambda=tradeoff
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_save_load.pt"))
        if fail:
            if fail == "img":
                curie_img = torch.rand_like(curie_img)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized image are different",
                )
            elif fail == "metric1":
                # this works with either rgb or grayscale images (though note
                # that SSIM just operates on each RGB channel independently,
                # which is probably not the right thing to do)
                metric = dis_ssim
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized optimized_metric are different"),
                )
            elif fail == "metric2":
                # this works with either rgb or grayscale images
                metric2 = rgb_mse
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized reference_metric are different"),
                )
            elif fail == "target":
                target = "max"
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized minmax are different",
                )
            elif fail == "tradeoff":
                tradeoff = 10
                expectation = pytest.raises(
                    ValueError,
                    match=(
                        "Saved and initialized metric_tradeoff_lambda are different"
                    ),
                )
            mad_copy = po.synth.MADCompetition(
                curie_img,
                metric,
                metric2,
                target,
                metric_tradeoff_lambda=tradeoff,
            )
            with expectation:
                mad_copy.load(
                    op.join(tmp_path, "test_mad_save_load.pt"),
                    map_location=DEVICE,
                )
        else:
            mad_copy = po.synth.MADCompetition(
                curie_img,
                metric,
                metric2,
                target,
                metric_tradeoff_lambda=tradeoff,
            )
            mad_copy.load(
                op.join(tmp_path, "test_mad_save_load.pt"), map_location=DEVICE
            )
            # check that can resume
            mad_copy.synthesize(max_iter=5, store_progress=True)
        if rgb:
            # since this is a fixture, get this back to a grayscale image
            curie_img = curie_img.mean(1, True)

    @pytest.mark.parametrize("optimizer", ["Adam", None, "Scheduler"])
    def test_optimizer_opts(self, curie_img, optimizer):
        mad = po.synth.MADCompetition(
            curie_img,
            po.metric.mse,
            lambda *args: 1 - po.metric.ssim(*args),
            "min",
        )
        scheduler = None
        if optimizer == "Adam" or optimizer == "Scheduler":
            optimizer = torch.optim.Adam([mad.mad_image])
            if optimizer == "Scheduler":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        mad.synthesize(max_iter=5, optimizer=optimizer, scheduler=scheduler)

    @pytest.mark.parametrize(
        "metric", [po.metric.mse, ModuleMetric(), NonModuleMetric()]
    )
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    def test_to(self, curie_img, metric, to_type):
        mad = po.synth.MADCompetition(curie_img, metric, po.tools.optim.l2_norm, "min")
        mad.synthesize(max_iter=5)
        if to_type == "dtype":
            mad.to(torch.float64)
            assert mad.initial_image.dtype == torch.float64
            assert mad.image.dtype == torch.float64
            assert mad.mad_image.dtype == torch.float64
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == "device" and DEVICE.type != "cpu":
            mad.to("cpu")
        # initial_image doesn't get used anywhere after init, so check it like
        # this
        mad.initial_image - mad.image
        mad.mad_image - mad.image
        mad.synthesize(max_iter=5)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    def test_map_location(self, curie_img, tmp_path):
        curie_img = curie_img
        mad = po.synth.MADCompetition(
            curie_img, po.metric.mse, po.tools.optim.l2_norm, "min"
        )
        mad.synthesize(max_iter=4, store_progress=True)
        mad.save(op.join(tmp_path, "test_mad_map_location.pt"))
        curie_img = curie_img.to("cpu")
        mad_copy = po.synth.MADCompetition(
            curie_img, po.metric.mse, po.tools.optim.l2_norm, "min"
        )
        assert mad_copy.image.device.type == "cpu"
        mad_copy.load(op.join(tmp_path, "test_mad_map_location.pt"), map_location="cpu")
        assert mad_copy.mad_image.device.type == "cpu"
        mad_copy.synthesize(max_iter=4, store_progress=True)

    # MAD can accept multiple images on the batch dimension, but the metrics
    # must return a single number. This means, effectively, that we can do
    # synthesis for e.g., video metrics, but that we cannot synthesize several
    # images in parallel
    def test_batch_synthesis(self, curie_img, einstein_img):
        img = torch.cat([curie_img, einstein_img], dim=0)
        mad = po.synth.MADCompetition(
            img,
            lambda *args: po.metric.mse(*args).mean(),
            po.tools.optim.l2_norm,
            "min",
        )
        mad.synthesize(max_iter=10)
        assert mad.mad_image.shape == img.shape, (
            "MAD image should have the same shape as input!"
        )

    @pytest.mark.parametrize("store_progress", [True, 2, 3])
    def test_store_rep(self, einstein_img, store_progress):
        mad = po.synth.MADCompetition(einstein_img, po.metric.mse, dis_ssim, "min")
        max_iter = 3
        if store_progress == 3:
            max_iter = 6
        mad.synthesize(max_iter=max_iter, store_progress=store_progress)
        assert len(mad.saved_mad_image) == np.ceil(max_iter / store_progress), (
            "Didn't end up with enough saved mad after first synth!"
        )
        assert len(mad.losses) == max_iter, (
            "Didn't end up with enough losses after first synth!"
        )
        # these have a +1 because we calculate them during initialization as
        # well (so we know our starting point).
        assert len(mad.optimized_metric_loss) == max_iter + 1, (
            "Didn't end up with enough optimized metric losses after first synth!"
        )
        assert len(mad.reference_metric_loss) == max_iter + 1, (
            "Didn't end up with enough reference metric losses after first synth!"
        )
        mad.synthesize(max_iter=max_iter, store_progress=store_progress)
        assert len(mad.saved_mad_image) == np.ceil(2 * max_iter / store_progress), (
            "Didn't end up with enough saved mad after second synth!"
        )
        assert len(mad.losses) == 2 * max_iter, (
            "Didn't end up with enough losses after second synth!"
        )
        assert len(mad.optimized_metric_loss) == 2 * max_iter + 1, (
            "Didn't end up with enough optimized metric losses after second synth!"
        )
        assert len(mad.reference_metric_loss) == 2 * max_iter + 1, (
            "Didn't end up with enough reference metric losses after second synth!"
        )

    def test_continue(self, einstein_img):
        mad = po.synth.MADCompetition(einstein_img, po.metric.mse, dis_ssim, "min")
        mad.synthesize(max_iter=3, store_progress=True)
        mad.synthesize(max_iter=3, store_progress=True)

    def test_nan_loss(self, einstein_img):
        # clone to prevent NaN from showing up in other tests
        img = einstein_img.clone()
        mad = po.synth.MADCompetition(img, po.metric.mse, dis_ssim, "min")
        mad.synthesize(max_iter=5)
        mad.image[..., 0, 0] = torch.nan
        with pytest.raises(ValueError, match="Found a NaN in loss during optimization"):
            mad.synthesize(max_iter=1)

    def test_change_precision_save_load(self, einstein_img, tmp_path):
        # Identity model doesn't change when you call .to() with a dtype
        # (unlike those models that have weights) so we use it here
        mad = po.synth.MADCompetition(einstein_img, po.metric.mse, dis_ssim, "min")
        mad.synthesize(max_iter=5)
        mad.to(torch.float64)
        assert mad.mad_image.dtype == torch.float64, "dtype incorrect!"
        mad.save(op.join(tmp_path, "test_change_prec_save_load.pt"))
        mad_copy = po.synth.MADCompetition(
            einstein_img.to(torch.float64), po.metric.mse, dis_ssim, "min"
        )
        mad_copy.load(op.join(tmp_path, "test_change_prec_save_load.pt"))
        mad_copy.synthesize(max_iter=5)
        assert mad_copy.mad_image.dtype == torch.float64, "dtype incorrect!"

    def test_stop_criterion(self, einstein_img):
        # checking that this hits the criterion and stops early, so set seed
        # for reproducibility
        po.tools.set_seed(0)
        mad = po.synth.MADCompetition(einstein_img, po.metric.mse, dis_ssim, "min")
        mad.synthesize(max_iter=15, stop_criterion=1e-3, stop_iters_to_check=5)
        assert abs(mad.losses[-5] - mad.losses[-1]) < 1e-3, (
            "Didn't stop when hit criterion!"
        )
        assert abs(mad.losses[-6] - mad.losses[-2]) > 1e-3, (
            "Stopped after hit criterion!"
        )
