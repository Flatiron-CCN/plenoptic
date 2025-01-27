"""Functions to validate synthesis inputs."""

import itertools
import warnings
from collections.abc import Callable

import torch
from torch import Tensor


def validate_input(
    input_tensor: Tensor,
    no_batch: bool = False,
    allowed_range: tuple[float, float] | None = None,
):
    """Determine whether input_tensor tensor can be used for synthesis.

    In particular, this function:

    - Checks if input_tensor has a float or complex dtype

    - Checks if input_tensor is 4d.

    - If ``no_batch`` is True, check whether ``input_tensor.shape[0] != 1``

    - If ``allowed_range`` is not None, check whether all values of
     ``input_tensor`` lie
      within the specified range.

    If any of the above fail, a ``ValueError`` is raised.

    Parameters
    ----------
    input_tensor
        The tensor to validate.
    no_batch
        If True, raise a ValueError if the batch dimension of ``input_tensor``
        is greater
        than 1.
    allowed_range
        If not None, ensure that all values of ``input_tensor`` lie within
        allowed_range.

    """
    # validate dtype
    if input_tensor.dtype not in [
        torch.float16,
        torch.complex32,
        torch.float32,
        torch.complex64,
        torch.float64,
        torch.complex128,
    ]:
        raise TypeError(
            "Only float or complex dtypes are"
            + f" allowed but got type {input_tensor.dtype}"
        )
    if input_tensor.ndimension() != 4:
        n_batch = 1 if no_batch else "n_batch"
        # numpy raises ValueError when operands cannot be broadcast together,
        # so it seems reasonable here
        raise ValueError(
            f"input_tensor must be torch.Size([{n_batch}, n_channels, "
            f"im_height, im_width]) but got shape {input_tensor.size()}"
        )
    if no_batch and input_tensor.shape[0] != 1:
        # numpy raises ValueError when operands cannot be broadcast together,
        # so it seems reasonable here
        raise ValueError("input_tensor batch dimension must be 1.")
    if allowed_range is not None:
        if allowed_range[0] >= allowed_range[1]:
            raise ValueError(
                "allowed_range[0] must be strictly less than"
                f" allowed_range[1], but got {allowed_range}"
            )
        if (
            input_tensor.min() < allowed_range[0]
            or input_tensor.max() > allowed_range[1]
        ):
            raise ValueError(
                f"input_tensor range must lie within {allowed_range}, but got"
                f" {(input_tensor.min().item(), input_tensor.max().item())}"
            )


def validate_model(
    model: torch.nn.Module,
    image_shape: tuple[int, int, int, int] | None = None,
    image_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    """Determine whether model can be used for sythesis.

    In particular, this function checks the following (with their associated
    errors raised):

    - If ``model`` adds a gradient to an input tensor, which implies that some
      of it is learnable (``ValueError``).

    - If ``model`` returns a tensor when given a tensor, failure implies that
      not all computations are done using torch (``ValueError``).

    - If ``model`` strips gradient from an input with gradient attached
      (``ValueError``).

    - If ``model`` casts an input tensor to something else and returns it to a
      tensor before returning it (``ValueError``).

    - If ``model`` changes the precision of the input tensor (``TypeError``).

    - If ``model`` returns a 3d or 4d output when given a 4d input
      (``ValueError``).

    - If ``model`` changes the device of the input (``RuntimeError``).

    Finally, we check if ``model`` is in training mode and raise a warning
    if so. Note that this is different from having learnable parameters,
    see ``pytorch docs <https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc>``_

    Parameters
    ----------
    model
        The model to validate.
    image_shape
        Some models (e.g., the steerable pyramid) can only accept inputs of a
        certain shape. If that's the case for ``model``, use this to
        specify the expected shape. If None, we use an image of shape
        (1,1,16,16)
    image_dtype
        What dtype to validate against.
    device
        What device to place test image on.

    See also
    --------
    remove_grad
        Helper function for detaching all parameters (in place).

    """
    if image_shape is None:
        image_shape = (1, 1, 16, 16)
    test_img = torch.rand(
        image_shape, dtype=image_dtype, requires_grad=False, device=device
    )
    try:
        if model(test_img).requires_grad:
            raise ValueError(
                "model adds gradient to input, at least one of its parameters"
                " is learnable. Try calling plenoptic.tools.remove_grad()"
                " on it."
            )
    # in particular, numpy arrays lack requires_grad attribute
    except AttributeError:
        raise ValueError(
            "model does not return a torch.Tensor object -- are you sure all"
            " computations are performed using torch?"
        )
    test_img.requires_grad_()
    try:
        if not model(test_img).requires_grad:
            raise ValueError(
                "model strips gradient from input, do you detach it somewhere?"
            )
    # this gets raised if something tries to cast a tensor with requires_grad
    # to an array, which can happen explicitly or if they try to use a numpy /
    # scipy / etc function. This gets reached (rather than the first
    # AttributeError) if they cast it to an array in the middle of forward()
    # and then try to cast it back to a tensor
    except RuntimeError:
        raise ValueError(
            "model tries to cast the input into something other than"
            " torch.Tensor object -- are you sure all computations are"
            " performed using torch?"
        )
    if image_dtype in [torch.float16, torch.complex32]:
        allowed_dtypes = [torch.float16, torch.complex32]
    elif image_dtype in [torch.float32, torch.complex64]:
        allowed_dtypes = [torch.float32, torch.complex64]
    elif image_dtype in [torch.float64, torch.complex128]:
        allowed_dtypes = [torch.float64, torch.complex128]
    else:
        raise TypeError(
            f"Only float or complex dtypes are allowed but got type {image_dtype}"
        )
    if model(test_img).dtype not in allowed_dtypes:
        raise TypeError("model changes precision of input, don't do that!")
    if model(test_img).ndimension() not in [3, 4]:
        raise ValueError(
            "When given a 4d input, model output must be three- or"
            " four-dimensional but had {model(test_img).ndimension()}"
            " dimensions instead!"
        )
    if model(test_img).device != test_img.device:
        # pytorch device errors are RuntimeErrors
        raise RuntimeError("model changes device of input, don't do that!")
    if hasattr(model, "training") and model.training:
        warnings.warn(
            "model is in training mode, you probably want to call eval()"
            " to switch to evaluation mode"
        )


def validate_coarse_to_fine(
    model: torch.nn.Module,
    image_shape: tuple[int, int, int, int] | None = None,
    device: str | torch.device = "cpu",
):
    """Determine whether a model can be used for coarse-to-fine synthesis.

    In particular, this function checks the following (with associated errors):

    - Whether ``model`` has a ``scales`` attribute (``AttributeError``).

    - Whether ``model.forward`` accepts a ``scales`` keyword argument (``TypeError``).

    - Whether the output of ``model.forward`` changes shape when the ``scales``
      keyword argument is set (``ValueError``).

    Parameters
    ----------
    model
        The model to validate.
    image_shape
        Some models (e.g., the steerable pyramid) can only accept inputs of a
        certain shape. If that's the case for ``model``, use this to
        specify the expected shape. If None, we use an image of shape
        (1,1,16,16)
    device
        Which device to place the test image on.

    """
    warnings.warn(
        "Validating whether model can work with coarse-to-fine synthesis --"
        " this can take a while!"
    )
    msg = "and therefore we cannot do coarse-to-fine synthesis"
    if not hasattr(model, "scales"):
        raise AttributeError(f"model has no scales attribute {msg}")
    if image_shape is None:
        image_shape = (1, 1, 16, 16)
    test_img = torch.rand(image_shape, device=device)
    model_output_shape = model(test_img).shape
    for len_val in range(1, len(model.scales)):
        for sc in itertools.combinations(model.scales, len_val):
            try:
                if model_output_shape == model(test_img, scales=sc).shape:
                    raise ValueError(
                        "Output of model forward method doesn't change"
                        " shape when scales keyword arg is set to {sc} {msg}"
                    )
            except TypeError:
                raise TypeError(
                    f"model forward method does not accept scales argument {sc} {msg}"
                )


def validate_metric(
    metric: torch.nn.Module | Callable[[Tensor, Tensor], Tensor],
    image_shape: tuple[int, int, int, int] | None = None,
    image_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    """Determines whether a metric can be used for MADCompetition synthesis.

    In particular, this functions checks the following (with associated
    exceptions):

    - Whether ``metric`` is callable and accepts two 4d tensors as input
      (``TypeError``).

    - Whether ``metric`` returns a scalar when called with two 4d tensors as
      input (``ValueError``).

    - Whether ``metric`` returns a value less than 5e-7 when with two identical
      4d tensors as input (``ValueError``). (This threshold was chosen because
      1-SSIM of two identical images is 5e-8 on GPU).

    Parameters
    ----------
    metric
        The metric to validate.
    image_shape
        Some models (e.g., the steerable pyramid) can only accept inputs of a
        certain shape. If that's the case for ``model``, use this to
        specify the expected shape. If None, we use an image of shape
        (1,1,16,16)
    image_dtype
        What dtype to validate against.
    device
        What device to place the test images on.

    """
    if image_shape is None:
        image_shape = (1, 1, 16, 16)
    test_img = torch.rand(image_shape, dtype=image_dtype, device=device)
    try:
        same_val = metric(test_img, test_img).item()
    except TypeError:
        raise TypeError("metric should be callable and accept two 4d tensors as input")
    # as of torch 2.0.0, this is a RuntimeError (a Tensor with X elements
    # cannot be converted to Scalar); previously it was a ValueError (only one
    # element tensors can be converted to Python scalars)
    except (ValueError, RuntimeError):
        raise ValueError(
            "metric should return a scalar value but"
            + f" output had shape {metric(test_img, test_img).shape}"
        )
    # on gpu, 1-SSIM of two identical images is 5e-8, so we use a threshold
    # of 5e-7 to check for zero
    if same_val > 5e-7:
        raise ValueError(
            "metric should return <= 5e-7 on"
            + f" two identical images but got {same_val}"
        )
    # this is hard to test
    for i in range(20):
        second_test_img = torch.rand_like(test_img)
        if metric(test_img, second_test_img).item() < 0:
            raise ValueError("metric should always return non-negative numbers!")


def remove_grad(model: torch.nn.Module):
    """Detach all parameters and buffers of model (in place)."""
    for p in model.parameters():
        if p.requires_grad:
            p.detach_()
    for p in model.buffers():
        if p.requires_grad:
            p.detach_()
