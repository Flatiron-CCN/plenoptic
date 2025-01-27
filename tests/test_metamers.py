# necessary to avoid issues with animate:
# https://github.com/matplotlib/matplotlib/issues/10287/
import matplotlib

matplotlib.use("agg")
import os.path as op

import numpy as np
import pytest
import torch

import plenoptic as po
from conftest import DEVICE


# in order for pickling to work with functions, they must be defined at top of
# module: https://stackoverflow.com/a/36995008
def custom_loss(x1, x2):
    return (x1 - x2).sum()


class TestMetamers:
    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("loss_func", ["mse", "l2", "custom"])
    @pytest.mark.parametrize(
        "fail", [False, "img", "model", "loss", "range_penalty", "dtype"]
    )
    @pytest.mark.parametrize("range_penalty", [0.1, 0])
    def test_save_load(
        self, einstein_img, model, loss_func, fail, range_penalty, tmp_path
    ):
        if loss_func == "mse":
            loss = po.tools.optim.mse
        elif loss_func == "l2":
            loss = po.tools.optim.l2_norm
        elif loss_func == "custom":
            loss = custom_loss
        met = po.synth.Metamer(
            einstein_img,
            model,
            loss_function=loss,
            range_penalty_lambda=range_penalty,
        )
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_save_load.pt"))
        if fail:
            if fail == "img":
                einstein_img = torch.rand_like(einstein_img)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized image are different",
                )
            elif fail == "model":
                model = po.simul.Gaussian(30).to(DEVICE)
                po.tools.remove_grad(model)
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized target_representation are different"),
                )
            elif fail == "loss":
                loss = po.metric.ssim
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized loss_function are different",
                )
            elif fail == "range_penalty":
                range_penalty = 0.5
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized range_penalty_lambda are different"),
                )
            elif fail == "dtype":
                einstein_img = einstein_img.to(torch.float64)
                # need to create new instance of model, because otherwise the
                # version with doubles as weights will persist for other tests
                model = po.simul.LinearNonlinear((31, 31)).to(DEVICE)
                po.tools.remove_grad(model)
                model.to(torch.float64)
                expectation = pytest.raises(
                    RuntimeError, match="Attribute image has different dtype"
                )
            met_copy = po.synth.Metamer(
                einstein_img,
                model,
                loss_function=loss,
                range_penalty_lambda=range_penalty,
            )
            with expectation:
                met_copy.load(
                    op.join(tmp_path, "test_metamer_save_load.pt"),
                    map_location=DEVICE,
                )
        else:
            met_copy = po.synth.Metamer(
                einstein_img,
                model,
                loss_function=loss,
                range_penalty_lambda=range_penalty,
            )
            met_copy.load(
                op.join(tmp_path, "test_metamer_save_load.pt"),
                map_location=DEVICE,
            )
            for k in [
                "image",
                "saved_metamer",
                "metamer",
                "target_representation",
            ]:
                if not getattr(met, k).allclose(getattr(met_copy, k), rtol=1e-2):
                    raise ValueError(
                        f"Something went wrong with saving and loading! {k} not"
                        " the same"
                    )
            # check loss functions correctly saved
            met_loss = met.loss_function(
                met.model(met.metamer), met.target_representation
            )
            met_copy_loss = met_copy.loss_function(
                met.model(met.metamer), met_copy.target_representation
            )
            if not torch.allclose(met_loss, met_copy_loss, rtol=1e-2):
                raise ValueError(
                    "Loss function not properly saved! Before saving was"
                    f" {met_loss}, after loading was {met_copy_loss}"
                )
            # check that can resume
            met_copy.synthesize(
                max_iter=4,
                store_progress=True,
            )

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    @pytest.mark.parametrize("store_progress", [True, 2, 3])
    def test_store_rep(self, einstein_img, model, store_progress):
        metamer = po.synth.Metamer(einstein_img, model)
        max_iter = 3
        if store_progress == 3:
            max_iter = 6
        metamer.synthesize(max_iter=max_iter, store_progress=store_progress)
        assert len(metamer.saved_metamer) == np.ceil(max_iter / store_progress), (
            "Didn't end up with enough saved metamer after first synth!"
        )
        assert len(metamer.losses) == max_iter, (
            "Didn't end up with enough losses after first synth!"
        )
        metamer.synthesize(max_iter=max_iter, store_progress=store_progress)
        assert len(metamer.saved_metamer) == np.ceil(2 * max_iter / store_progress), (
            "Didn't end up with enough saved metamer after second synth!"
        )
        assert len(metamer.losses) == 2 * max_iter, (
            "Didn't end up with enough losses after second synth!"
        )

    @pytest.mark.parametrize(
        "model", ["frontend.LinearNonlinear.nograd"], indirect=True
    )
    def test_continue(self, einstein_img, model):
        metamer = po.synth.Metamer(einstein_img, model)
        metamer.synthesize(max_iter=3, store_progress=True)
        metamer.synthesize(max_iter=3, store_progress=True)

    @pytest.mark.parametrize("model", ["SPyr"], indirect=True)
    @pytest.mark.parametrize("coarse_to_fine", ["separate", "together"])
    def test_coarse_to_fine(self, einstein_img, model, coarse_to_fine, tmp_path):
        metamer = po.synth.MetamerCTF(
            einstein_img, model, coarse_to_fine=coarse_to_fine
        )
        metamer.synthesize(
            max_iter=5,
            stop_iters_to_check=1,
            change_scale_criterion=10,
            ctf_iters_to_check=1,
        )
        assert len(metamer.scales_finished) > 0, "Didn't actually switch scales!"

        metamer.save(op.join(tmp_path, "test_metamer_ctf.pt"))
        metamer_copy = po.synth.MetamerCTF(
            einstein_img, model, coarse_to_fine=coarse_to_fine
        )
        metamer_copy.load(op.join(tmp_path, "test_metamer_ctf.pt"), map_location=DEVICE)
        # check the ctf-related attributes all saved correctly
        for k in [
            "coarse_to_fine",
            "scales",
            "scales_loss",
            "scales_timing",
            "scales_finished",
        ]:
            if not getattr(metamer, k) == (getattr(metamer_copy, k)):
                raise ValueError(
                    f"Something went wrong with saving and loading! {k} not the same"
                )
        # check we can resume
        metamer.synthesize(
            max_iter=5,
            stop_iters_to_check=1,
            change_scale_criterion=10,
            ctf_iters_to_check=1,
        )

    @pytest.mark.parametrize("model", ["NLP"], indirect=True)
    @pytest.mark.parametrize("optimizer", ["Adam", None, "Scheduler"])
    def test_optimizer(self, curie_img, model, optimizer):
        met = po.synth.Metamer(curie_img, model)
        scheduler = None
        if optimizer == "Adam" or optimizer == "Scheduler":
            optimizer = torch.optim.Adam([met.metamer])
            if optimizer == "Scheduler":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        met.synthesize(max_iter=5, optimizer=optimizer, scheduler=scheduler)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["Identity"], indirect=True)
    def test_map_location(self, curie_img, model, tmp_path):
        curie_img = curie_img.to(DEVICE)
        model.to(DEVICE)
        met = po.synth.Metamer(curie_img, model)
        met.synthesize(max_iter=4, store_progress=True)
        met.save(op.join(tmp_path, "test_metamer_map_location.pt"))
        # calling load with map_location effectively switches everything
        # over to that device
        met_copy = po.synth.Metamer(curie_img, model)
        met_copy.load(
            op.join(tmp_path, "test_metamer_map_location.pt"),
            map_location="cpu",
        )
        assert met_copy.metamer.device.type == "cpu"
        assert met_copy.image.device.type == "cpu"
        met_copy.synthesize(max_iter=4, store_progress=True)

    @pytest.mark.parametrize("model", ["Identity", "NonModule"], indirect=True)
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    def test_to(self, curie_img, model, to_type):
        met = po.synth.Metamer(curie_img, model)
        met.synthesize(max_iter=5)
        if to_type == "dtype":
            met.to(torch.float16)
            assert met.image.dtype == torch.float16
            assert met.metamer.dtype == torch.float16
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == "device" and DEVICE.type != "cpu":
            met.to("cpu")
        met.metamer - met.image
        met.synthesize(max_iter=5)

    # this determines whether we mix across channels or treat them separately,
    # both of which are supported
    @pytest.mark.parametrize("model", ["ColorModel", "Identity"], indirect=True)
    def test_multichannel(self, model, color_img):
        met = po.synth.Metamer(color_img, model)
        met.synthesize(max_iter=5)
        assert met.metamer.shape == color_img.shape, (
            "Metamer image should have the same shape as input!"
        )

    # this determines whether we mix across batches (e.g., a video model) or
    # treat them separately, both of which are supported
    @pytest.mark.parametrize("model", ["VideoModel", "Identity"], indirect=True)
    def test_multibatch(self, model, einstein_img, curie_img):
        img = torch.cat([curie_img, einstein_img], dim=0)
        met = po.synth.Metamer(img, model)
        met.synthesize(max_iter=5)
        assert met.metamer.shape == img.shape, (
            "Metamer image should have the same shape as input!"
        )

    # we assume that the target representation has no gradient attached, so
    # doublecheck that (validate_model should ensure this)
    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_rep_no_grad(self, model, einstein_img):
        met = po.synth.Metamer(einstein_img, model)
        assert met.target_representation.grad is None, (
            "Target representation has a gradient attached, how?"
        )
        met.synthesize(max_iter=5)
        assert met.target_representation.grad is None, (
            "Target representation has a gradient attached, how?"
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_nan_loss(self, model, einstein_img):
        # clone to prevent NaN from showing up in other tests
        img = einstein_img.clone()
        met = po.synth.Metamer(img, model)
        met.synthesize(max_iter=5)
        met.target_representation[..., 0, 0] = torch.nan
        with pytest.raises(ValueError, match="Found a NaN in loss during optimization"):
            met.synthesize(max_iter=1)

    @pytest.mark.parametrize("model", ["Identity"], indirect=True)
    def test_change_precision_save_load(self, model, einstein_img, tmp_path):
        # Identity model doesn't change when you call .to() with a dtype
        # (unlike those models that have weights) so we use it here
        met = po.synth.Metamer(einstein_img, model)
        met.synthesize(max_iter=5)
        met.to(torch.float64)
        assert met.metamer.dtype == torch.float64, "dtype incorrect!"
        met.save(op.join(tmp_path, "test_metamer_change_prec_save_load.pt"))
        met_copy = po.synth.Metamer(einstein_img.to(torch.float64), model)
        met_copy.load(op.join(tmp_path, "test_metamer_change_prec_save_load.pt"))
        met_copy.synthesize(max_iter=5)
        assert met_copy.metamer.dtype == torch.float64, "dtype incorrect!"

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_stop_criterion(self, einstein_img, model):
        # checking that this hits the criterion and stops early, so set seed
        # for reproducibility
        po.tools.set_seed(0)
        met = po.synth.Metamer(einstein_img, model)
        # takes different numbers of iter to converge on GPU and CPU
        met.synthesize(max_iter=35, stop_criterion=1e-5, stop_iters_to_check=5)
        assert abs(met.losses[-5] - met.losses[-1]) < 1e-5, (
            "Didn't stop when hit criterion!"
        )
        assert abs(met.losses[-6] - met.losses[-2]) > 1e-5, (
            "Stopped after hit criterion!"
        )
