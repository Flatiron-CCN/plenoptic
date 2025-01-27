from contextlib import nullcontext as does_not_raise
from math import pi

import einops
import numpy as np
import pytest
import scipy.ndimage
import torch
from numpy.random import randint

import plenoptic as po
from conftest import DEVICE, IMG_DIR


class TestData:
    def test_load_images_fail(self):
        with pytest.raises(ValueError, match="All images must be the same shape"):
            po.load_images(
                [
                    IMG_DIR / "256" / "einstein.pgm",
                    IMG_DIR / "mixed" / "bubbles.png",
                ]
            )


class TestSignal:
    def test_polar_amplitude_zero(self):
        a = torch.rand(10, device=DEVICE) * -1
        b = po.tools.rescale(torch.randn(10, device=DEVICE), -pi / 2, pi / 2)

        with pytest.raises(ValueError) as _:
            _, _ = po.tools.polar_to_rectangular(a, b)

    def test_coordinate_identity_transform_rectangular(self):
        dims = (10, 5, 256, 256)
        x = torch.randn(dims, device=DEVICE)
        y = torch.randn(dims, device=DEVICE)

        z = po.tools.polar_to_rectangular(
            *po.tools.rectangular_to_polar(torch.complex(x, y))
        )

        assert torch.linalg.vector_norm((x - z.real).flatten(), ord=2) < 1e-3
        assert torch.linalg.vector_norm((y - z.imag).flatten(), ord=2) < 1e-3

    def test_coordinate_identity_transform_polar(self):
        dims = (10, 5, 256, 256)

        # ensure vec len a is non-zero by adding .1 and then re-normalizing
        a = torch.rand(dims, device=DEVICE) + 0.1
        a = a / a.max()
        b = po.tools.rescale(torch.randn(dims, device=DEVICE), -pi / 2, pi / 2)

        A, B = po.tools.rectangular_to_polar(po.tools.polar_to_rectangular(a, b))

        assert torch.linalg.vector_norm((a - A).flatten(), ord=2) < 1e-3
        assert torch.linalg.vector_norm((b - B).flatten(), ord=2) < 1e-3

    @pytest.mark.parametrize("n", range(1, 15))
    def test_autocorrelation(self, n):
        x = po.tools.make_synthetic_stimuli().to(DEVICE)
        x_centered = x - x.mean((2, 3), keepdim=True)
        a = po.tools.autocorrelation(x_centered)
        a = po.tools.center_crop(a, n)

        # autocorr with zero delay is variance
        assert (
            torch.abs(torch.var(x, dim=(2, 3)) - a[..., n // 2, n // 2]) < 1e-5
        ).all()

        # autocorr can be computed in signal domain directly with roll
        h = randint(-(n // 2), ((n + 1) // 2))
        assert (
            torch.abs(
                (x_centered * torch.roll(x_centered, h, dims=2)).sum((2, 3))
                / (x.shape[-2] * x.shape[-1])
                - a[..., n // 2 + h, n // 2]
            )
            < 1e-5
        ).all()

        w = randint(-(n // 2), ((n + 1) // 2))
        assert (
            torch.abs(
                (x_centered * torch.roll(x_centered, w, dims=3)).sum((2, 3))
                / (x.shape[-2] * x.shape[-1])
                - a[..., n // 2, n // 2 + w]
            )
            < 1e-5
        ).all()

    @pytest.mark.parametrize("size_A", [1, 3])
    @pytest.mark.parametrize("size_B", [1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_add_noise(self, einstein_img, size_A, size_B, dtype):
        A = einstein_img.repeat(size_A, 1, 1, 1).to(dtype)
        B = size_B * [4]
        if size_A != size_B and size_A != 1 and size_B != 1:
            with pytest.raises(Exception):
                po.tools.add_noise(A, B)
        else:
            assert po.tools.add_noise(A, B).shape[0] == max(size_A, size_B)

    @pytest.mark.parametrize("factor", [0.5, 1, 1.5, 2, 1.1])
    @pytest.mark.parametrize("img_size", [256, 128, 200])
    def test_expand(self, factor, img_size, einstein_img):
        einstein_img = einstein_img.clone()[..., :img_size]
        if int(factor * img_size) != factor * img_size:
            expectation = pytest.raises(
                ValueError, match="factor \* x.shape\[-1\] must be"
            )
        elif int(factor * einstein_img.shape[-2]) != factor * einstein_img.shape[-2]:
            expectation = pytest.raises(
                ValueError, match="factor \* x.shape\[-2\] must be"
            )
        elif factor <= 1:
            expectation = pytest.raises(
                ValueError, match="factor must be strictly greater"
            )
        else:
            expectation = does_not_raise()
        with expectation:
            expanded = po.tools.expand(einstein_img, factor)
            np.testing.assert_equal(
                expanded.shape[-2:],
                [factor * s for s in einstein_img.shape[-2:]],
            )

    @pytest.mark.parametrize("factor", [0.5, 1, 1.5, 2, 1.1])
    @pytest.mark.parametrize("img_size", [256, 128, 200])
    def test_shrink(self, factor, img_size, einstein_img):
        einstein_img = einstein_img.clone()[..., :img_size]
        if int(img_size / factor) != img_size / factor:
            expectation = pytest.raises(
                ValueError, match="x.shape\[-1\]/factor must be"
            )
        elif int(einstein_img.shape[-2] / factor) != einstein_img.shape[-2] / factor:
            expectation = pytest.raises(
                ValueError, match="x.shape\[-2\]/factor must be"
            )
        elif factor <= 1:
            expectation = pytest.raises(
                ValueError, match="factor must be strictly greater"
            )
        else:
            expectation = does_not_raise()
        with expectation:
            shrunk = po.tools.shrink(einstein_img, factor)
            np.testing.assert_equal(
                shrunk.shape[-2:],
                [s / factor for s in einstein_img.shape[-2:]],
            )

    @pytest.mark.parametrize("batch_channel", [[1, 3], [2, 1], [2, 3]])
    def test_shrink_batch_channel(self, batch_channel, einstein_img):
        shrunk = po.tools.shrink(einstein_img.repeat((*batch_channel, 1, 1)), 2)
        size = batch_channel + [s / 2 for s in einstein_img.shape[-2:]]
        np.testing.assert_equal(shrunk.shape, size)

    @pytest.mark.parametrize("batch_channel", [[1, 3], [2, 1], [2, 3]])
    def test_expand_batch_channel(self, batch_channel, einstein_img):
        expanded = po.tools.expand(einstein_img.repeat((*batch_channel, 1, 1)), 2)
        size = batch_channel + [2 * s for s in einstein_img.shape[-2:]]
        np.testing.assert_equal(expanded.shape, size)

    @pytest.mark.parametrize("factor", [1.5, 2])
    @pytest.mark.parametrize("img", ["curie", "einstein", "metal", "nuts"])
    def test_expand_shrink(self, img, factor):
        # expand then shrink will be the same as the original image, up to this
        # fudge factor
        img = po.load_images(IMG_DIR / "256" / f"{img}.pgm").to(DEVICE)
        modified = po.tools.shrink(po.tools.expand(img, factor), factor)
        torch.testing.assert_close(img, modified, atol=2e-2, rtol=1e-6)

    @pytest.mark.parametrize("phase", [0, np.pi / 2, np.pi])
    def test_modulate_phase_correlation(self, phase):
        # here we create an image that has sinusoids at two frequencies, with
        # some phase offset. Because their frequencies are an octave apart,
        # they will show up in neighboring scales of the steerable pyramid
        # coefficients. Based on their phase offset, the coefficients should
        # either be correlated, uncorrelated, or anti-correlated, which is only
        # recoverable after using expand and doubling the phase of the lower
        # frequency one (this trick is used in th PS texture model)
        X = torch.arange(256).unsqueeze(1).repeat(1, 256) / 256 * 2 * torch.pi
        X = X.unsqueeze(0).unsqueeze(0)
        X = torch.sin(8 * X) + torch.sin(16 * X + phase)

        pyr = po.simul.SteerablePyramidFreq(X.shape[-2:], is_complex=True)
        pyr_coeffs = pyr(X)
        a = pyr_coeffs[(3, 2)]
        b = pyr_coeffs[(2, 2)]
        a = po.tools.expand(a, 2) / 4
        a = po.tools.modulate_phase(a, 2)

        # this is the correlation as computed in the PS texture model, which is
        # where modulate phase is used
        corr = einops.einsum(a.real, b.real, "b c h w, b c h w -> b c")
        corr = corr / (torch.mul(*a.shape[-2:])) / (a.std() * b.std())

        tgt_corr = {0: 0.4999, np.pi / 2: 0, np.pi: -0.4999}[phase]

        np.testing.assert_allclose(corr, tgt_corr, rtol=1e-5, atol=1e-5)

    def test_modulate_phase_noreal(self):
        X = torch.arange(256).unsqueeze(1).repeat(1, 256) / 256 * 2 * torch.pi
        X = X.unsqueeze(0).unsqueeze(0)

        with pytest.raises(TypeError, match="x must be a complex-valued tensor"):
            po.tools.modulate_phase(X)

    @pytest.mark.parametrize("batch_channel", [(1, 3), (2, 1), (2, 3)])
    def test_modulate_phase_batch_channel(self, batch_channel):
        X = torch.arange(256).unsqueeze(1).repeat(1, 256) / 256 * 2 * torch.pi
        X = X.unsqueeze(0).unsqueeze(0).repeat((*batch_channel, 1, 1))
        X = torch.sin(8 * X) + torch.sin(16 * X)

        pyr = po.simul.SteerablePyramidFreq(X.shape[-2:], is_complex=True)
        pyr_coeffs = pyr(X)
        a = pyr_coeffs[(3, 2)]
        a = po.tools.expand(a, 2) / 4
        a = po.tools.modulate_phase(a, 2)

        # shape should be preferred
        np.testing.assert_equal(a.shape[:2], batch_channel)

        # because the signal is just repeated along the batch and channel dims,
        # modulated version should be too (ensures we're not mixing batch or
        # channel)
        np.testing.assert_array_equal(a, a.roll(1, 1))
        np.testing.assert_array_equal(a, a.roll(1, 0))


class TestStats:
    def test_stats(self):
        torch.manual_seed(0)
        B, D = 32, 512
        x = torch.randn(B, D)
        m = torch.mean(x, dim=1, keepdim=True)
        v = po.tools.variance(x, mean=m, dim=1, keepdim=True)
        assert (
            torch.abs(v - torch.var(x, dim=1, keepdim=True, unbiased=False)) < 1e-5
        ).all()
        po.tools.skew(x, mean=m, var=v, dim=1)
        k = po.tools.kurtosis(x, mean=m, var=v, dim=1)
        assert torch.abs(k.mean() - 3) < 1e-1

        k = po.tools.kurtosis(torch.rand(B, D), dim=1)
        assert k.mean() < 3

        scale = 2
        exp_samples1 = -scale * torch.log(torch.rand(B, D))
        exp_samples2 = -scale * torch.log(torch.rand(B, D))
        lap_samples = exp_samples1 - exp_samples2
        k = po.tools.kurtosis(lap_samples, dim=1)
        assert k.mean() > 3

    @pytest.mark.parametrize("batch_channel", [(1, 1), (1, 3), (2, 1), (2, 3)])
    def test_var_multidim(self, batch_channel):
        B, D = 32, 512
        x = torch.randn(*batch_channel, B, D)
        var = po.tools.variance(x, dim=(-1, -2))
        np.testing.assert_equal(var.shape, batch_channel)

    @pytest.mark.parametrize("batch_channel", [(1, 1), (1, 3), (2, 1), (2, 3)])
    def test_skew_multidim(self, batch_channel):
        B, D = 32, 512
        x = torch.randn(*batch_channel, B, D)
        skew = po.tools.skew(x, dim=(-1, -2))
        np.testing.assert_equal(skew.shape, batch_channel)

    @pytest.mark.parametrize("batch_channel", [(1, 1), (1, 3), (2, 1), (2, 3)])
    def test_kurt_multidim(self, batch_channel):
        B, D = 32, 512
        x = torch.randn(*batch_channel, B, D)
        kurt = po.tools.kurtosis(x, dim=(-1, -2))
        np.testing.assert_equal(kurt.shape, batch_channel)


class TestDownsampleUpsample:
    @pytest.mark.parametrize("odd", [0, 1])
    @pytest.mark.parametrize("size", [9, 10, 11, 12])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_filter(self, odd, size, dtype):
        img = torch.zeros([1, 1, 24 + odd, 25], device=DEVICE, dtype=dtype)
        img[0, 0, 12, 12] = 1
        filt = np.zeros([size, size + 1])
        filt[5, 5] = 1
        filt = scipy.ndimage.gaussian_filter(filt, sigma=1)
        filt = torch.as_tensor(filt, dtype=dtype, device=DEVICE)
        img_down = po.tools.correlate_downsample(img, filt=filt)
        img_up = po.tools.upsample_convolve(img_down, odd=(odd, 1), filt=filt)
        assert np.unravel_index(img_up.cpu().numpy().argmax(), img_up.shape) == (
            0,
            0,
            12,
            12,
        )

        img_down = po.tools.blur_downsample(img)
        img_up = po.tools.upsample_blur(img_down, odd=(odd, 1))
        assert np.unravel_index(img_up.cpu().numpy().argmax(), img_up.shape) == (
            0,
            0,
            12,
            12,
        )

    def test_multichannel(self):
        img = torch.randn([10, 3, 24, 25], device=DEVICE, dtype=torch.float32)
        filt = torch.randn([5, 5], device=DEVICE, dtype=torch.float32)
        img_down = po.tools.correlate_downsample(img, filt=filt)
        img_up = po.tools.upsample_convolve(img_down, odd=(0, 1), filt=filt)
        assert img_up.shape == img.shape

        img_down = po.tools.blur_downsample(img)
        img_up = po.tools.upsample_blur(img_down, odd=(0, 1))
        assert img_up.shape == img.shape


class TestValidate:
    # https://docs.pytest.org/en/4.6.x/example/parametrize.html#parametrizing-conditional-raising
    @pytest.mark.parametrize(
        "shape,expectation",
        [
            ((1, 1, 16, 16), does_not_raise()),
            ((1, 3, 16, 16), does_not_raise()),
            ((2, 1, 16, 16), does_not_raise()),
            ((2, 3, 16, 16), does_not_raise()),
            (
                (1, 1, 1, 16, 16),
                pytest.raises(ValueError, match="input_tensor must be torch.Size"),
            ),
            (
                (1, 16, 16),
                pytest.raises(ValueError, match="input_tensor must be torch.Size"),
            ),
            (
                (16, 16),
                pytest.raises(ValueError, match="input_tensor must be torch.Size"),
            ),
        ],
    )
    def test_input_shape(self, shape, expectation):
        img = torch.rand(*shape)
        with expectation:
            po.tools.validate.validate_input(img)

    def test_input_no_batch(self):
        img = torch.rand(2, 1, 16, 16)
        with pytest.raises(ValueError, match="input_tensor batch dimension must be 1"):
            po.tools.validate.validate_input(img, no_batch=True)

    @pytest.mark.parametrize(
        "minmax,expectation",
        [
            (
                "min",
                pytest.raises(ValueError, match="input_tensor range must lie within"),
            ),
            (
                "max",
                pytest.raises(ValueError, match="input_tensor range must lie within"),
            ),
            (
                "range",
                pytest.raises(
                    ValueError,
                    match=r"allowed_range\[0\] must be strictly less",
                ),
            ),
        ],
    )
    def test_input_allowed_range(self, minmax, expectation):
        img = torch.rand(1, 1, 16, 16)
        allowed_range = (0, 1)
        if minmax == "min":
            img -= 1
        elif minmax == "max":
            img += 1
        elif minmax == "range":
            allowed_range = (1, 0)
        with expectation:
            po.tools.validate.validate_input(img, allowed_range=allowed_range)

    @pytest.mark.parametrize("model", ["frontend.OnOff"], indirect=True)
    def test_model_learnable(self, model):
        with pytest.raises(ValueError, match="model adds gradient to input"):
            po.tools.validate.validate_model(model, device=DEVICE)

    def test_model_numpy_comp(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, img):
                return np.fft.fft(img)

        model = TestModel()
        with pytest.raises(
            ValueError, match="model does not return a torch.Tensor object"
        ):
            # don't pass device here because the model just uses numpy, which
            # only works on cpu
            po.tools.validate.validate_model(model)

    def test_model_detach(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, img):
                return img.detach()

        model = TestModel()
        with pytest.raises(ValueError, match="model strips gradient from input"):
            po.tools.validate.validate_model(model, device=DEVICE)

    def test_model_numpy_and_back(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, img):
                return torch.from_numpy(np.fft.fft(img))

        model = TestModel()
        with pytest.raises(
            ValueError,
            match="model tries to cast the input into something other",
        ):
            # don't pass device here because the model just uses numpy, which
            # only works on cpu
            po.tools.validate.validate_model(model)

    def test_model_precision(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, img):
                return img.to(torch.float16)

        model = TestModel()
        with pytest.raises(TypeError, match="model changes precision of input"):
            po.tools.validate.validate_model(model, device=DEVICE)

    @pytest.mark.parametrize("direction", ["squeeze", "unsqueeze"])
    def test_model_output_dim(self, direction):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, img):
                if direction == "squeeze":
                    return img.squeeze()
                elif direction == "unsqueeze":
                    return img.unsqueeze(0)

        model = TestModel()
        with pytest.raises(ValueError, match="When given a 4d input, model output"):
            po.tools.validate.validate_model(model, device=DEVICE)

    @pytest.mark.skipif(DEVICE.type == "cpu", reason="Only makes sense to test on cuda")
    def test_model_device(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, img):
                return img.to("cpu")

        model = TestModel()
        with pytest.raises(RuntimeError, match="model changes device of input"):
            po.tools.validate.validate_model(model, device=DEVICE)

    @pytest.mark.parametrize("model", ["ColorModel"], indirect=True)
    def test_model_image_shape(self, model):
        img_shape = (1, 3, 16, 16)
        po.tools.validate.validate_model(model, image_shape=img_shape, device=DEVICE)

    def test_validate_ctf_scales(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, img):
                return img

        model = TestModel()
        with pytest.raises(AttributeError, match="model has no scales attribute"):
            po.tools.validate.validate_coarse_to_fine(model, device=DEVICE)

    def test_validate_ctf_arg(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scales = [0, 1, 2]

            def forward(self, img):
                return img

        model = TestModel()
        with pytest.raises(
            TypeError,
            match="model forward method does not accept scales argument",
        ):
            po.tools.validate.validate_coarse_to_fine(model, device=DEVICE)

    def test_validate_ctf_shape(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scales = [0, 1, 2]

            def forward(self, img, scales=[]):
                return img

        model = TestModel()
        with pytest.raises(
            ValueError,
            match="Output of model forward method doesn't change shape",
        ):
            po.tools.validate.validate_coarse_to_fine(model, device=DEVICE)

    def test_validate_ctf_pass(self):
        model = po.simul.PortillaSimoncelli((256, 256)).to(DEVICE)
        po.tools.validate.validate_coarse_to_fine(
            model, image_shape=(1, 1, *model.image_shape), device=DEVICE
        )

    # Metric validation tests
    def test_validate_metric_inputs(self):
        def identity_metric(x):
            return x

        with pytest.raises(TypeError, match="metric should be callable and accept two"):
            po.tools.validate.validate_metric(identity_metric, device=DEVICE)

    def test_validate_metric_output_shape(self):
        def difference_metric(x, y):
            return x - y

        with pytest.raises(
            ValueError, match="metric should return a scalar value but output"
        ):
            po.tools.validate.validate_metric(difference_metric, device=DEVICE)

    def test_validate_metric_identical(self):
        def mean_metric(x, y):
            return (x + y).mean()

        with pytest.raises(
            ValueError, match="metric should return <= 5e-7 on two identical"
        ):
            po.tools.validate.validate_metric(mean_metric, device=DEVICE)

    def test_validate_metric_nonnegative(self):
        po.tools.set_seed(0)

        def sum_metric(x, y):
            return (x - y).sum()

        with pytest.raises(
            ValueError, match="metric should always return non-negative"
        ):
            po.tools.validate.validate_metric(sum_metric, device=DEVICE)

    @pytest.mark.parametrize("model", ["frontend.OnOff.nograd"], indirect=True)
    def test_remove_grad(self, model):
        po.tools.validate.validate_model(model, device=DEVICE)


class TestOptim:
    def test_penalize_range_above(self):
        img = 0.5 * torch.ones((1, 1, 4, 4))
        img[..., 0, :] = 2
        assert po.tools.optim.penalize_range(img).item() == 4

    def test_penalize_range_below(self):
        img = 0.5 * torch.ones((1, 1, 4, 4))
        img[..., 0, :] = -1
        assert po.tools.optim.penalize_range(img).item() == 4


class TestPolarImages:
    def test_polar_angle_clockwise(self):
        ang = po.tools.polar_angle(100, direction="clockwise")
        idx = np.argmin((ang - np.pi / 2) ** 2)
        assert np.unravel_index(idx, (100, 100))[0] > 50, (
            "pi/2 should be in bottom half of image!"
        )
        idx = np.argmin((ang + np.pi / 2) ** 2)
        assert np.unravel_index(idx, (100, 100))[0] < 50, (
            "pi/2 should be in top half of image!"
        )

    def test_polar_angle_counterclockwise(self):
        ang = po.tools.polar_angle(100, direction="counter-clockwise")
        idx = np.argmin((ang - np.pi / 2) ** 2)
        assert np.unravel_index(idx, (100, 100))[0] < 50, (
            "pi/2 should be in top half of image!"
        )
        idx = np.argmin((ang + np.pi / 2) ** 2)
        assert np.unravel_index(idx, (100, 100))[0] > 50, (
            "pi/2 should be in bottom half of image!"
        )

    def test_polar_angle_direction(self):
        with pytest.raises(ValueError, match="direction must be one of"):
            po.tools.polar_angle(100, direction="-clockwise")
