"""Helper functions for testing."""

import pathlib
import re

import numpy as np
import pyrtools as pt
import torch

import plenoptic as po
from test_models import TestPortillaSimoncelli


def update_ps_synthesis_test_file(torch_version: str | None = None):
    """Create new test file for test_models.test_ps_synthesis().

    We cannot guarantee perfect reproducibility across pytorch versions, but we
    can guarantee it within versions. Therefore, we need to periodically create
    new files to test our synthesis outputs against. This helper function does
    that, but note that you NEED to examine the resulting metamer manually to
    ensure it still looks good.

    After generating this file and checking it looks good, upload it to the OSF
    plenoptic-files project: https://osf.io/ts37w/files/osfstorage, then update
    test_models.get_portilla_simoncelli_synthesize_filename and
    fetch.REGISTRY_URLS and fetch.REGISTRY

    Parameters
    ----------
    torch_version
        The version of pytorch for which we should grab the corresponding test
        file. If None, we use the installed version.

    Returns
    -------
    met : po.synth.Metamer
        Metamer object for inspection

    """
    ps_synth_file = po.data.fetch_data(
        "portilla_simoncelli_synthesize_torch_v1.12.0_ps-refactor-2.npz"
    )
    print(f"Loading from {ps_synth_file}")

    with np.load(ps_synth_file) as f:
        im = f["im"]
        im_init = f["im_init"]
        im_synth = f["im_synth"]
        rep_synth = f["rep_synth"]

    met = TestPortillaSimoncelli().test_ps_synthesis(ps_synth_file, False)

    torch_v = torch.__version__.split("+")[0]
    file_name_parts = re.findall(
        "(.*portilla_simoncelli_synthesize)(_gpu)?(_torch_v)?([0-9.]*)(_ps-refactor)?(-2)?.npz",
        ps_synth_file.name,
    )[0]
    output_file_name = (
        "".join(file_name_parts[:2]) + f"_torch_v{torch_v}{file_name_parts[-1]}.npz"
    )
    output = po.to_numpy(met.metamer).squeeze()
    rep = po.to_numpy(met.model(met.metamer)).squeeze()
    try:
        np.testing.assert_allclose(output, im_synth.squeeze(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(rep, rep_synth.squeeze(), rtol=1e-4, atol=1e-4)
        print(
            "Current synthesis same as saved version, so not saving current synthesis."
        )
    # only do all this if the tests would've failed
    except AssertionError:
        print(f"Saving at {output_file_name}")
        np.savez(
            output_file_name,
            im=im,
            im_init=im_init,
            im_synth=output,
            rep_synth=rep,
        )
        fig = pt.imshow(
            [output, im_synth.squeeze()],
            title=[f"New metamer (torch {torch_v})", "Old metamer"],
        )
        fig.savefig(output_file_name.replace(".npz", ".png"))
    return met


def update_ps_torch_output(save_dir):
    """Create new test files for test_ps_torch_output().

    Unlike update_ps_synthesis_test_file(), this only needs to get updated if we
    change the model.

    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_scales = [1, 2, 3, 4]
    n_orientations = [2, 3, 4]
    spatial_corr_width = range(3, 10)
    IMG_DIR = po.data.fetch_data("test_images.tar.gz")
    im_names = ["curie", "einstein", "metal", "nuts"]
    ims = po.load_images([IMG_DIR / "256" / f"{im}.pgm" for im in im_names])
    for scale in n_scales:
        for ori in n_orientations:
            for width in spatial_corr_width:
                for im, name in zip(ims, im_names):
                    ps = po.simul.PortillaSimoncelli(
                        im.shape[-2:],
                        n_scales=scale,
                        n_orientations=ori,
                        spatial_corr_width=width,
                    ).to(torch.float64)
                    output = po.to_numpy(ps(im.unsqueeze(0)))
                    fname = (
                        save_dir / f"{name}_scales-{scale}_ori-{ori}_spat-{width}.npy"
                    )
                    np.save(fname, output)
    print(
        f"All outputs have been saved in directory {save_dir},"
        + f"now go to {save_dir.parent} "
        f"and run `tar czf {save_dir.name}"
        + f" --directory={save_dir.with_suffix('.tar.gz').name}/ .`"
    )


def update_ps_scales(save_path):
    """Create new test files for test_ps_torch_output().

    Unlike update_ps_synthesis_test_file(), this only needs to get updated if we
    change the model.

    """
    save_path = pathlib.Path(save_path)
    if save_path.suffix != ".npz":
        raise ValueError(f"save_path must have suffix .npz but got {save_path.suffix}!")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_scales = [1, 2, 3, 4]
    n_orientations = [2, 3, 4]
    spatial_corr_width = range(3, 10)
    output = {}
    for scale in n_scales:
        for ori in n_orientations:
            for width in spatial_corr_width:
                ps = po.simul.PortillaSimoncelli(
                    (256, 256),
                    n_scales=scale,
                    n_orientations=ori,
                    spatial_corr_width=width,
                )
                key = f"scale-{scale}_ori-{ori}_width-{width}"
                output[key] = ps._representation_scales
    np.savez(save_path, **output)
