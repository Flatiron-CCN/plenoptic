import os.path as op
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

import plenoptic as po
from conftest import DEVICE


class TestSequences:
    def test_deviation_from_line_and_brownian_bridge(self):
        """this probabilistic test passes with high probability
        in high dimensions, but for reproducibility we
        set the seed manually."""
        torch.manual_seed(0)
        t = 2**6
        d = 2**14
        sqrt_d = int(np.sqrt(d))
        start = torch.randn(1, d).reshape(1, 1, sqrt_d, sqrt_d).to(DEVICE)
        stop = torch.randn(1, d).reshape(1, 1, sqrt_d, sqrt_d).to(DEVICE)
        b = po.tools.sample_brownian_bridge(start, stop, t, d**0.5)
        a, f = po.tools.deviation_from_line(b, normalize=True)
        assert torch.abs(a[t // 2] - 0.5) < 1e-2, f"{a[t // 2]}"
        assert torch.abs(f[t // 2] - 2**0.5 / 2) < 1e-2, f"{f[t // 2]}"

    @pytest.mark.parametrize("normalize", [True, False])
    def test_deviation_from_line_multichannel(self, normalize, einstein_img):
        einstein_img = einstein_img.repeat(1, 3, 1, 1)
        seq = po.tools.translation_sequence(einstein_img)
        dist_along, dist_from = po.tools.deviation_from_line(seq, normalize)
        assert dist_along.shape[0] == seq.shape[0], (
            "Distance along line has wrong number of transitions!"
        )
        assert dist_from.shape[0] == seq.shape[0], (
            "Distance from  line has wrong number of transitions!"
        )

    @pytest.mark.parametrize("n_steps", [1, 10])
    @pytest.mark.parametrize("max_norm", [0, 1, 10])
    @pytest.mark.parametrize("multichannel", [False, True])
    def test_brownian_bridge(
        self, einstein_img, curie_img, n_steps, multichannel, max_norm
    ):
        if multichannel:
            einstein_img = einstein_img.repeat(1, 3, 1, 1)
            curie_img = curie_img.repeat(1, 3, 1, 1)
        bridge = po.tools.sample_brownian_bridge(
            einstein_img, curie_img, n_steps, max_norm
        )
        assert bridge.shape == (
            n_steps + 1,
            *einstein_img.shape[1:],
        ), "sample_brownian_bridge returned a tensor of the wrong shape!"

    @pytest.mark.parametrize("fail", ["batch", "same_shape", "n_steps", "max_norm"])
    def test_brownian_bridge_fail(self, einstein_img, curie_img, fail):
        n_steps = 2
        max_norm = 1
        if fail == "batch":
            einstein_img = einstein_img.repeat(2, 1, 1, 1)
            curie_img = curie_img.repeat(2, 1, 1, 1)
            expectation = pytest.raises(
                ValueError, match="input_tensor batch dimension must be 1"
            )
        elif fail == "same_shape":
            # rand_like preserves DEVICE and dtype
            curie_img = torch.rand_like(curie_img)[..., :128, :128]
            expectation = pytest.raises(
                ValueError, match="start and stop must be same shape"
            )
        elif fail == "n_steps":
            n_steps = 0
            expectation = pytest.raises(ValueError, match="n_steps must be positive")
        elif fail == "max_norm":
            max_norm = -1
            expectation = pytest.raises(
                ValueError, match="max_norm must be non-negative"
            )
        with expectation:
            po.tools.sample_brownian_bridge(einstein_img, curie_img, n_steps, max_norm)

    @pytest.mark.parametrize("n_steps", [1, 10])
    @pytest.mark.parametrize("multichannel", [False, True])
    def test_straight_line(self, einstein_img, curie_img, n_steps, multichannel):
        if multichannel:
            einstein_img = einstein_img.repeat(1, 3, 1, 1)
            curie_img = curie_img.repeat(1, 3, 1, 1)
        line = po.tools.make_straight_line(einstein_img, curie_img, n_steps)
        assert line.shape == (
            n_steps + 1,
            *einstein_img.shape[1:],
        ), "make_straight_line returned a tensor of the wrong shape!"

    @pytest.mark.parametrize("fail", ["batch", "same_shape", "n_steps"])
    def test_straight_line_fail(self, einstein_img, curie_img, fail):
        n_steps = 2
        if fail == "batch":
            einstein_img = einstein_img.repeat(2, 1, 1, 1)
            curie_img = curie_img.repeat(2, 1, 1, 1)
            expectation = pytest.raises(
                ValueError, match="input_tensor batch dimension must be 1"
            )
        elif fail == "same_shape":
            # rand_like preserves DEVICE and dtype
            curie_img = torch.rand_like(curie_img)[..., :128, :128]
            expectation = pytest.raises(
                ValueError, match="start and stop must be same shape"
            )
        elif fail == "n_steps":
            n_steps = 0
            expectation = pytest.raises(ValueError, match="n_steps must be positive")
        with expectation:
            po.tools.make_straight_line(einstein_img, curie_img, n_steps)

    @pytest.mark.parametrize("n_steps", [0, 1, 10])
    @pytest.mark.parametrize("multichannel", [False, True])
    def test_translation_sequence(self, einstein_img, n_steps, multichannel):
        if n_steps == 0:
            expectation = pytest.raises(ValueError, match="n_steps must be positive")
        else:
            expectation = does_not_raise()
        if multichannel:
            einstein_img = einstein_img.repeat(1, 3, 1, 1)
        with expectation:
            shifted = po.tools.translation_sequence(einstein_img, n_steps)
            assert torch.equal(shifted[0], einstein_img[0]), (
                "somehow first frame changed!"
            )
            assert torch.equal(shifted[1, 0, :, 1], shifted[0, 0, :, 0]), (
                "wrong dimension was translated!"
            )

    @pytest.mark.parametrize(
        "func",
        [
            "make_straight_line",
            "translation_sequence",
            "sample_brownian_bridge",
            "deviation_from_line",
        ],
    )
    def test_preserve_device(self, einstein_img, func):
        kwargs = {}
        if func != "deviation_from_line":
            kwargs["n_steps"] = 5
            if func != "translation_sequence":
                kwargs["stop"] = torch.rand_like(einstein_img)
        seq = getattr(po.tools, func)(einstein_img, **kwargs)
        # kinda hacky -- deviation_from_line returns a tuple, all the others
        # return a 4d tensor. regardless seq[0] will be a tensor
        assert seq[0].device == einstein_img.device, f"{func} changed device!"


class TestGeodesic:
    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    @pytest.mark.parametrize("init", ["straight", "bridge"])
    @pytest.mark.parametrize("optimizer", [None, "SGD"])
    @pytest.mark.parametrize("n_steps", [5, 10])
    def test_texture(self, einstein_img_small, model, init, optimizer, n_steps):
        sequence = po.tools.translation_sequence(einstein_img_small, n_steps)
        moog = po.synth.Geodesic(sequence[:1], sequence[-1:], model, n_steps, init)
        if optimizer == "SGD":
            optimizer = torch.optim.SGD([moog._geodesic], lr=0.01)
        moog.synthesize(max_iter=5, optimizer=optimizer)
        po.synth.geodesic.plot_loss(moog)
        po.synth.geodesic.plot_deviation_from_line(moog, natural_video=sequence)
        moog.calculate_jerkiness()

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_endpoints_dont_change(self, einstein_small_seq, model):
        moog = po.synth.Geodesic(
            einstein_small_seq[:1],
            einstein_small_seq[-1:],
            model,
            5,
            "straight",
        )
        moog.synthesize(max_iter=5)
        assert torch.equal(moog.geodesic[0], einstein_small_seq[0]), (
            "Somehow first endpoint changed!"
        )
        assert torch.equal(moog.geodesic[-1], einstein_small_seq[-1]), (
            "Somehow last endpoint changed!"
        )
        assert not torch.equal(moog.pixelfade[1:-1], moog.geodesic[1:-1]), (
            "Somehow middle of geodesic didn't changed!"
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    @pytest.mark.parametrize(
        "fail",
        [False, "img_a", "img_b", "model", "n_steps", "init", "range_penalty"],
    )
    def test_save_load(self, einstein_small_seq, model, fail, tmp_path):
        img_a = einstein_small_seq[:1]
        img_b = einstein_small_seq[-1:]
        n_steps = 3
        init = "straight"
        range_penalty = 0
        moog = po.synth.Geodesic(
            img_a,
            img_b,
            model,
            n_steps,
            init,
            range_penalty_lambda=range_penalty,
        )
        moog.synthesize(max_iter=4)
        moog.save(op.join(tmp_path, "test_geodesic_save_load.pt"))
        if fail:
            if fail == "img_a":
                img_a = torch.rand_like(img_a)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized image_a are different",
                )
            elif fail == "img_b":
                img_b = torch.rand_like(img_b)
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized image_b are different",
                )
            elif fail == "model":
                model = po.simul.Gaussian(30).to(DEVICE)
                po.tools.remove_grad(model)
                expectation = pytest.raises(
                    ValueError,
                    match="objective_function on pixelfade of saved",
                )
            elif fail == "n_steps":
                n_steps = 5
                expectation = pytest.raises(
                    ValueError,
                    match="Saved and initialized n_steps are different",
                )
            elif fail == "init":
                init = "bridge"
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized initial_sequence are different"),
                )
            elif fail == "range_penalty":
                range_penalty = 0.5
                expectation = pytest.raises(
                    ValueError,
                    match=("Saved and initialized range_penalty_lambda are different"),
                )
            moog_copy = po.synth.Geodesic(
                img_a,
                img_b,
                model,
                n_steps,
                init,
                range_penalty_lambda=range_penalty,
            )
            with expectation:
                moog_copy.load(
                    op.join(tmp_path, "test_geodesic_save_load.pt"),
                    map_location=DEVICE,
                )
        else:
            moog_copy = po.synth.Geodesic(
                img_a,
                img_b,
                model,
                n_steps,
                init,
                range_penalty_lambda=range_penalty,
            )
            moog_copy.load(
                op.join(tmp_path, "test_geodesic_save_load.pt"),
                map_location=DEVICE,
            )
            for k in ["image_a", "image_b", "pixelfade", "geodesic"]:
                if not getattr(moog, k).allclose(getattr(moog_copy, k), rtol=1e-2):
                    raise ValueError(
                        "Something went wrong with saving and loading!"
                        f" {k} not the same"
                    )
            # check that can resume
            moog_copy.synthesize(max_iter=4)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    @pytest.mark.parametrize("model", ["Identity"], indirect=True)
    def test_map_location(self, einstein_small_seq, model, tmp_path):
        moog = po.synth.Geodesic(einstein_small_seq[:1], einstein_small_seq[-1:], model)
        moog.synthesize(max_iter=4, store_progress=True)
        moog.save(op.join(tmp_path, "test_geodesic_map_location.pt"))
        # calling load with map_location effectively switches everything
        # over to that device
        moog_copy = po.synth.Geodesic(
            einstein_small_seq[:1], einstein_small_seq[-1:], model
        )
        moog_copy.load(
            op.join(tmp_path, "test_geodesic_map_location.pt"),
            map_location="cpu",
        )
        assert moog_copy.geodesic.device.type == "cpu"
        assert moog_copy.image_a.device.type == "cpu"
        moog_copy.synthesize(max_iter=4, store_progress=True)

    @pytest.mark.parametrize("model", ["Identity"], indirect=True)
    @pytest.mark.parametrize("to_type", ["dtype", "device"])
    def test_to(self, einstein_small_seq, model, to_type):
        moog = po.synth.Geodesic(einstein_small_seq[:1], einstein_small_seq[-1:], model)
        moog.synthesize(max_iter=5)
        if to_type == "dtype":
            moog.to(torch.float16)
            assert moog.image_a.dtype == torch.float16
            assert moog.geodesic.dtype == torch.float16
        # can only run this one if we're on a device with CPU and GPU.
        elif to_type == "device" and DEVICE.type != "cpu":
            moog.to("cpu")
        moog.geodesic - moog.image_a

    @pytest.mark.parametrize("model", ["Identity"], indirect=True)
    def test_change_precision_save_load(self, einstein_small_seq, model, tmp_path):
        # Identity model doesn't change when you call .to() with a dtype
        # (unlike those models that have weights) so we use it here
        moog = po.synth.Geodesic(einstein_small_seq[:1], einstein_small_seq[-1:], model)
        moog.synthesize(max_iter=5)
        moog.to(torch.float64)
        assert moog.geodesic.dtype == torch.float64, "dtype incorrect!"
        moog.save(op.join(tmp_path, "test_change_prec_save_load.pt"))
        seq = einstein_small_seq.to(torch.float64)
        moog_copy = po.synth.Geodesic(seq[:1], seq[-1:], model)
        moog_copy.load(op.join(tmp_path, "test_change_prec_save_load.pt"))
        moog_copy.synthesize(max_iter=5)
        assert moog_copy.geodesic.dtype == torch.float64, "dtype incorrect!"

    # this determines whether we mix across channels or treat them separately,
    # both of which are supported
    @pytest.mark.parametrize("model", ["ColorModel", "Identity"], indirect=True)
    def test_multichannel(self, color_img, model):
        img = color_img[..., :64, :64]
        seq = po.tools.translation_sequence(img, 5)
        moog = po.synth.Geodesic(seq[:1], seq[-1:], model, 5)
        moog.synthesize(max_iter=5)
        assert moog.geodesic.shape[1:] == img.shape[1:], (
            "Geodesic image should have same number of channels, height, width"
            " shape as input!"
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    @pytest.mark.parametrize("func", ["objective_function", "calculate_jerkiness"])
    def test_funcs_external_tensor(self, einstein_small_seq, model, func):
        moog = po.synth.Geodesic(
            einstein_small_seq[:1], einstein_small_seq[-1:], model, 5
        )
        no_arg = getattr(moog, func)()
        arg_tensor = torch.rand_like(moog.geodesic)
        # calculate jerkiness requires tensor to have gradient attached
        # (because we use autodiff functions)
        if func == "calculate_jerkiness":
            arg_tensor.requires_grad_()
        with_arg = getattr(moog, func)(arg_tensor)
        assert not torch.equal(no_arg, with_arg), (
            f"{func} is not using the input tensor!"
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_continue(self, einstein_small_seq, model):
        moog = po.synth.Geodesic(
            einstein_small_seq[:1], einstein_small_seq[-1:], model, 5
        )
        moog.synthesize(max_iter=3, store_progress=True)
        moog.synthesize(max_iter=3, store_progress=True)

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_nan_loss(self, model, einstein_small_seq):
        # clone to prevent NaN from showing up in other tests
        seq = einstein_small_seq.clone()
        moog = po.synth.Geodesic(seq[:1], seq[-1:], model, 5)
        moog.synthesize(max_iter=5)
        moog.image_a[..., 0, 0] = torch.nan
        with pytest.raises(ValueError, match="Found a NaN in loss during optimization"):
            moog.synthesize(max_iter=1)

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    @pytest.mark.parametrize("store_progress", [True, 2, 3])
    def test_store_progress(self, einstein_small_seq, model, store_progress):
        moog = po.synth.Geodesic(
            einstein_small_seq[:1], einstein_small_seq[-1:], model, 5
        )
        max_iter = 3
        if store_progress == 3:
            max_iter = 6
        moog.synthesize(max_iter=max_iter, store_progress=store_progress)
        assert len(moog.step_energy) == np.ceil(max_iter / store_progress), (
            "Didn't end up with enough step_energy after first synth!"
        )
        assert len(moog.dev_from_line) == np.ceil(max_iter / store_progress), (
            "Didn't end up with enough dev_from_line after first synth!"
        )
        assert len(moog.losses) == max_iter, (
            "Didn't end up with enough losses after first synth!"
        )
        moog.synthesize(max_iter=max_iter, store_progress=store_progress)
        assert len(moog.step_energy) == np.ceil(2 * max_iter / store_progress), (
            "Didn't end up with enough step_energy after second synth!"
        )
        assert len(moog.dev_from_line) == np.ceil(2 * max_iter / store_progress), (
            "Didn't end up with enough dev_from_line after second synth!"
        )
        assert len(moog.losses) == 2 * max_iter, (
            "Didn't end up with enough losses after second synth!"
        )

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_stop_criterion(self, einstein_small_seq, model):
        # checking that this hits the criterion and stops early, so set seed
        # for reproducibility
        po.tools.set_seed(0)
        moog = po.synth.Geodesic(
            einstein_small_seq[:1], einstein_small_seq[-1:], model, 5
        )
        moog.synthesize(max_iter=10, stop_criterion=0.06, stop_iters_to_check=1)
        assert (abs(moog.pixel_change_norm[-1:]) < 0.06).all(), (
            "Didn't stop when hit criterion!"
        )
        assert (abs(moog.pixel_change_norm[:-1]) > 0.06).all(), (
            "Stopped after hit criterion!"
        )
