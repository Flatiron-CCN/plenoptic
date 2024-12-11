from contextlib import nullcontext as does_not_raise
from contextlib import suppress
from itertools import product

import numpy as np
import pyrtools as pt
import pytest
import torch

import plenoptic as po
from conftest import DEVICE, IMG_DIR
from plenoptic.tools.data import to_numpy

ALL_SPYRS = (
    [
        f"{h}-{o}-{c}-{d}-{tf}"
        for h, o, c, d, tf in product(
            ["auto", 1, 3, 4, 5],
            [1, 2, 3],
            [True, False],
            [True, False],
            [True, False],
        )
    ]
    + [
        # pyramid with order=0 can only be tight frame if it's not complex
        f"{h}-0-False-{d}-{tf}"
        for h, d, tf in product(
            ["auto", 1, 3, 4, 5],
            [True, False],
            [True, False],
        )
    ]
    + [
        f"{h}-0-True-{d}-False"
        for h, d in product(
            ["auto", 1, 3, 4, 5],
            [True, False],
        )
    ]
)


def check_pyr_coeffs(coeff_1, coeff_2, rtol=1e-3, atol=1e-3):
    """
    function that checks if two sets of pyramid coefficients are the same
    We set an absolute and relative tolerance and the following function checks if
    abs(coeff1-coeff2) <= atol + rtol*abs(coeff1)
    Inputs:
    coeff1: first dictionary of pyramid coefficients
    coeff2: second dictionary of pyramid coefficients
    Both coeffs must obviously have the same number of scales, orientations etc.
    """

    for k in coeff_1:
        if torch.is_tensor(coeff_1[k]):
            coeff_1_np = to_numpy(coeff_1[k].squeeze())
        else:
            coeff_1_np = coeff_1[k]
        if torch.is_tensor(coeff_2[k]):
            coeff_2_np = to_numpy(coeff_2[k].squeeze())
        else:
            coeff_2_np = coeff_2[k]

        np.testing.assert_allclose(coeff_1_np, coeff_2_np, rtol=rtol, atol=atol)


def check_band_energies(coeff_1, coeff_2, rtol=1e-4, atol=1e-4):
    """
    function that checks if the energy in each band of two pyramids are the same.
    We set an absolute and relative tolerance and the function checks for each band if
    abs(coeff_1-coeff_2) <= atol + rtol*abs(coeff_1)
    Args:
    coeff_1: first dictionary of torch tensors corresponding to each band
    coeff_2: second dictionary of torch tensors corresponding to each band
    """

    for i in range(len(coeff_1.items())):
        k1 = list(coeff_1.keys())[i]
        k2 = list(coeff_2.keys())[i]
        band_1 = to_numpy(coeff_1[k1])
        band_2 = to_numpy(coeff_2[k2])
        band_1 = band_1.squeeze()
        band_2 = band_2.squeeze()

        np.testing.assert_allclose(
            np.sum(np.abs(band_1) ** 2),
            np.sum(np.abs(band_2) ** 2),
            rtol=rtol,
            atol=atol,
        )


def check_parseval(im, coeff, rtol=1e-4, atol=0):
    """
    function that checks if the pyramid is parseval, i.e. energy of coeffs is
    the same as the energy in the original image.
    Args:
    input image: image stimulus as torch.Tensor
    coeff: dictionary of torch tensors corresponding to each band
    """
    total_band_energy = 0
    im_energy = np.sum(to_numpy(im) ** 2)
    for k, v in coeff.items():
        band = to_numpy(coeff[k])
        band = band.squeeze()

        total_band_energy += np.sum(np.abs(band) ** 2)

    np.testing.assert_allclose(total_band_energy, im_energy, rtol=rtol, atol=atol)


class TestSteerablePyramid:
    @pytest.fixture(
        scope="class",
        params=[
            f"{im}-{shape}"
            for im in ["einstein", "curie"]
            for shape in [None, 224, "128_1", "128_2"]
        ],
    )
    def img(self, request):
        im, shape = request.param.split("-")
        img = po.load_images(IMG_DIR / "256" / f"{im}.pgm").to(DEVICE)
        if shape == "224":
            img = img[..., :224, :224]
        elif shape == "128_1":
            img = img[..., :128, :]
        elif shape == "128_2":
            img = img[..., :128]
        return img

    @pytest.fixture(
        scope="class",
        params=[f"{shape}" for shape in [None, 224, "128_1", "128_2"]],
    )
    def multichannel_img(self, request):
        shape = request.param
        # use fixture for img and use color_wheel instead.
        img = po.load_images(IMG_DIR / "mixed" / "flowers.jpg", as_gray=False).to(
            DEVICE
        )
        if shape == "224":
            img = img[..., :224, :224]
        elif shape == "128_1":
            img = img[..., :128, :]
        elif shape == "128_2":
            img = img[..., :128]
        return img

    # WARNING: because this fixture requires the img fixture above, it should
    # only be used in tests that also use the img fixture. That is, tests where
    # you want to test both the einstein and curie images, as well as the
    # different sizes. Otherwise, this will generate a bunch of tests that use
    # the spyr with those strange shapes
    @pytest.fixture(scope="class")
    def spyr(self, img, request):
        height, order, is_complex, downsample, tightframe = request.param.split("-")

        with suppress(ValueError):
            # then height = 'auto' and that's fine
            height = int(height)
        # need to use eval to get from 'False' (string) to False (bool);
        # bool('False') == True, annoyingly enough
        pyr = po.simul.SteerablePyramidFreq(
            img.shape[-2:],
            height,
            int(order),
            is_complex=eval(is_complex),
            downsample=eval(downsample),
            tight_frame=eval(tightframe),
        )
        pyr.to(DEVICE)
        return pyr

    @pytest.fixture(scope="class")
    def spyr_multi(self, multichannel_img, request):
        height, order, is_complex, downsample, tightframe = request.param.split("-")

        with suppress(ValueError):
            # then height = 'auto' and that's fine
            height = int(height)
        # need to use eval to get from 'False' (string) to False (bool);
        # bool('False') == True, annoyingly enough
        pyr = po.simul.SteerablePyramidFreq(
            multichannel_img.shape[-2:],
            height,
            int(order),
            is_complex=eval(is_complex),
            downsample=eval(downsample),
            tight_frame=eval(tightframe),
        )
        pyr.to(DEVICE)
        return pyr

    # can't use one of the spyr fixtures here because we need to instantiate separately
    # for each of these shapes
    @pytest.mark.parametrize("height", ["auto", 1, 3, 4, 5])
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize(
        "im_shape",
        [None, (255, 255), (256, 128), (128, 256), (255, 256), (256, 255)],
    )
    def test_pyramid(self, basic_stim, height, order, is_complex, im_shape):
        if im_shape is not None:
            basic_stim = basic_stim[..., : im_shape[0], : im_shape[1]]
        expectation = does_not_raise()
        if (order == 0 and is_complex) or (
            im_shape is not None and any([im_shape[0] % 2, im_shape[1] % 2])
        ):
            expectation = pytest.warns(
                Warning, match="Reconstruction will not be perfect"
            )
        with expectation:
            spc = po.simul.SteerablePyramidFreq(
                basic_stim.shape[-2:],
                height=height,
                order=order,
                is_complex=is_complex,
            ).to(DEVICE)
        spc(basic_stim)

    @pytest.mark.parametrize(
        "spyr",
        [
            f"{h}-{o}-{c}-{d}-True"
            for h, o, c, d in product(
                ["auto", 1, 2, 3], [1, 2, 3], [True, False], [True, False]
            )
        ]
        + [
            # pyramid with order=0 can only be tight frame if it's not complex
            f"{h}-0-False-{d}-True"
            for h, d in product(["auto", 1, 2, 3], [True, False])
        ],
        indirect=True,
    )
    def test_tight_frame(self, img, spyr):
        pyr_coeffs = spyr.forward(img)
        check_parseval(img, pyr_coeffs)

    @pytest.mark.parametrize("height", ["auto", 1, 2, 3])
    @pytest.mark.parametrize("downsample", [True, False])
    def test_not_tight_frame(self, height, downsample):
        with pytest.raises(ValueError, match="cannot be tight-frame"):
            po.simul.SteerablePyramidFreq(
                (256, 256),
                height,
                0,
                is_complex=True,
                downsample=downsample,
                tight_frame=True,
            )

    @pytest.mark.parametrize(
        "spyr",
        [
            f"{h}-{o}-{c}-True-{t}"
            for h, o, c, t in product(
                [3, 4, 5], [1, 2, 3], [True, False], [True, False]
            )
        ]
        + [
            # pyramid with order=0 can only be tight frame if it's not complex
            f"{h}-0-False-True-{t}"
            for h, t in product([3, 4, 5], [True, False])
        ]
        + [
            # pyramid with order=0 can only be tight frame if it's not complex
            f"{h}-0-True-True-False"
            for h in [3, 4, 5]
        ],
        indirect=True,
    )
    def test_not_downsample(self, img, spyr):
        pyr_coeffs = spyr.forward(img)
        # need to add 1 because our heights are 0-indexed (i.e., the lowest
        # height has k[0]==0)
        height = max([k[0] for k in pyr_coeffs if isinstance(k[0], int)]) + 1
        # couldn't come up with a way to get this with fixtures, so we
        # instantiate it each time.
        spyr_not_downsample = po.simul.SteerablePyramidFreq(
            img.shape[-2:],
            height,
            spyr.order,
            is_complex=spyr.is_complex,
            downsample=False,
            tight_frame=spyr.tight_frame,
        )
        spyr_not_downsample.to(DEVICE)
        pyr_coeffs_nd = spyr_not_downsample.forward(img)
        check_band_energies(pyr_coeffs, pyr_coeffs_nd)

    @pytest.mark.parametrize(
        "scales",
        [
            [0],
            [1],
            [0, 1, 2],
            [2],
            None,
            ["residual_highpass", "residual_lowpass"],
            ["residual_highpass", 0, 1, "residual_lowpass"],
        ],
    )
    @pytest.mark.parametrize(
        "spyr",
        [
            f"{h}-{o}-{c}-False-False"
            for h, o, c in product([3, 4, 5], [0, 1, 2, 3], [True, False])
        ],
        indirect=True,
    )
    def test_pyr_to_tensor(self, img, spyr, scales, rtol=1e-12, atol=1e-12):
        pyr_coeff_dict = spyr.forward(img, scales=scales)
        split_complex = [True, False] if spyr.is_complex else [False]

        for val in split_complex:
            pyr_tensor, pyr_info = spyr.convert_pyr_to_tensor(
                pyr_coeff_dict, split_complex=val
            )
            pyr_coeff_dict2 = spyr.convert_tensor_to_pyr(pyr_tensor, *pyr_info)
            check_pyr_coeffs(pyr_coeff_dict, pyr_coeff_dict2, rtol, atol)

    @pytest.mark.parametrize(
        "spyr",
        [
            f"{h}-{o}-{c}-True-False"
            for h, o, c in product([3, 4, 5], [0, 1, 2, 3], [True, False])
        ],
        indirect=True,
    )
    def test_torch_vs_numpy_pyr(self, img, spyr):
        torch_spc = spyr.forward(img)
        # need to add 1 because our heights are 0-indexed (i.e., the lowest
        # height has k[0]==0)
        height = max([k[0] for k in torch_spc if isinstance(k[0], int)]) + 1
        pyrtools_sp = pt.pyramids.SteerablePyramidFreq(
            to_numpy(img.squeeze()),
            height=height,
            order=spyr.order,
            is_complex=spyr.is_complex,
        )
        pyrtools_spc = pyrtools_sp.pyr_coeffs
        check_pyr_coeffs(pyrtools_spc, torch_spc)

    @pytest.mark.parametrize(
        "spyr",
        ALL_SPYRS,
        indirect=True,
    )
    def test_complete_recon(self, img, spyr):
        pyr_coeffs = spyr.forward(img)
        recon = to_numpy(spyr.recon_pyr(pyr_coeffs))
        # reconstruction is bad in this context
        if spyr.order == 0 and spyr.is_complex:
            np.testing.assert_allclose(recon, to_numpy(img), atol=5e-1, rtol=1e-1)
        else:
            np.testing.assert_allclose(recon, to_numpy(img), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "spyr_multi",
        ALL_SPYRS,
        indirect=True,
    )
    def test_complete_recon_multi(self, multichannel_img, spyr_multi):
        pyr_coeffs = spyr_multi.forward(multichannel_img)
        recon = to_numpy(spyr_multi.recon_pyr(pyr_coeffs))
        # reconstruction is bad in this context
        if spyr_multi.order == 0 and spyr_multi.is_complex:
            np.testing.assert_allclose(
                recon, to_numpy(multichannel_img), atol=5e-1, rtol=1e-1
            )
        else:
            np.testing.assert_allclose(
                recon, to_numpy(multichannel_img), rtol=1e-4, atol=1e-4
            )

    @pytest.mark.parametrize(
        "spyr",
        [
            f"{h}-{o}-{c}-{d}-{tf}"
            for h, o, c, d, tf in product(
                ["auto"], [3], [True, False], [True, False], [True, False]
            )
        ],
        indirect=True,
    )
    def test_partial_recon(self, img, spyr):
        pyr_coeffs = spyr.forward(img)
        # need to add 1 because our heights are 0-indexed (i.e., the lowest
        # height has k[0]==0)
        height = max([k[0] for k in pyr_coeffs if isinstance(k[0], int)]) + 1
        pt_spyr = pt.pyramids.SteerablePyramidFreq(
            to_numpy(img.squeeze()),
            height=height,
            order=spyr.order,
            is_complex=spyr.is_complex,
        )
        recon_levels = [[0], [1, 3], [1, 3, 4]]
        recon_bands = [[1], [1, 3]]
        for levels, bands in product(["all"] + recon_levels, ["all"] + recon_bands):
            po_recon = to_numpy(spyr.recon_pyr(pyr_coeffs, levels, bands).squeeze())
            pt_recon = pt_spyr.recon_pyr(levels, bands)
            np.testing.assert_allclose(po_recon, pt_recon, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "spyr",
        [
            f"{h}-{o}-{c}-True-False"
            for h, o, c in product(["auto", 1, 3, 4], [0, 1, 2, 3], [True, False])
        ],
        indirect=True,
    )
    def test_recon_match_pyrtools(self, img, spyr, rtol=1e-6, atol=1e-6):
        # this should fail if and only if test_complete_recon does, but
        # may as well include it just in case
        pyr_coeffs = spyr.forward(img)
        # need to add 1 because our heights are 0-indexed (i.e., the lowest
        # height has k[0]==0)
        height = max([k[0] for k in pyr_coeffs if isinstance(k[0], int)]) + 1
        pt_pyr = pt.pyramids.SteerablePyramidFreq(
            to_numpy(img.squeeze()),
            height=height,
            order=spyr.order,
            is_complex=spyr.is_complex,
        )
        po_recon = po.to_numpy(spyr.recon_pyr(pyr_coeffs).squeeze())
        pt_recon = pt_pyr.recon_pyr()
        np.testing.assert_allclose(po_recon, pt_recon, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "scales",
        [
            [0],
            [4],
            [0, 1, 2],
            [0, 3, 4],
            ["residual_highpass", "residual_lowpass"],
            ["residual_highpass", 0, 1, "residual_lowpass"],
        ],
    )
    @pytest.mark.parametrize(
        "spyr",
        [f"auto-3-{c}-{d}-False" for c, d in product([True, False], [True, False])],
        indirect=True,
    )
    def test_scales_arg(self, img, spyr, scales):
        pyr_coeffs = spyr.forward(img)
        reduced_pyr_coeffs = spyr.forward(img, scales)
        for k, v in reduced_pyr_coeffs.items():
            if (v != pyr_coeffs[k]).any():
                raise Exception(
                    "Reduced pyr_coeffs should be same as original, but at"
                    f" least key {k} is not"
                )

        # recon_pyr should always fail
        with pytest.raises(Exception):
            spyr.recon_pyr()
        with pytest.raises(Exception):
            spyr.recon_pyr(scales)

    @pytest.mark.parametrize("height", range(-1, 8))
    def test_height_values(self, img, height):
        if height < 0:
            expectation = pytest.raises(
                ValueError, match="Height must be a non-negative int"
            )
        elif height > np.log2(min(img.shape[-2:])) - 2:
            expectation = pytest.raises(
                ValueError, match="Cannot build pyramid higher than"
            )
        else:
            expectation = does_not_raise()
        with expectation:
            pyr = po.simul.SteerablePyramidFreq(img.shape[-2:], height=height).to(
                DEVICE
            )
            pyr(img)

    @pytest.mark.parametrize("order", range(-1, 17))
    def test_order_values(self, img, order):
        if order in [-1, 16]:
            expectation = pytest.raises(
                ValueError, match="order must be an integer in the range"
            )
        else:
            expectation = does_not_raise()
        with expectation:
            pyr = po.simul.SteerablePyramidFreq(img.shape[-2:], order=order).to(DEVICE)
            pyr(img)

    @pytest.mark.parametrize("order", range(0, 16))
    def test_buffers(self, order):
        pyr = po.simul.SteerablePyramidFreq((256, 256), order=order)
        buffers = [k for k, _ in pyr.named_buffers()]
        names = ["lo0mask", "hi0mask"]
        for s in range(pyr.num_scales):
            names.extend(
                [
                    f"_himasks_scale_{s}",
                    f"_lomasks_scale_{s}",
                    f"_anglemasks_scale_{s}",
                    f"_anglemasks_recon_scale_{s}",
                ]
            )
        assert len(buffers) == len(
            names
        ), "pyramid doesn't have the right number of buffers!"
        assert set(buffers) == set(names), "pyramid doesn't have the right buffers!"
