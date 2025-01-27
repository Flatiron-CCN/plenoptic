"""abstract synthesis super-class."""

import abc
import warnings

import numpy as np
import torch


class Synthesis(abc.ABC):
    r"""Abstract super-class for synthesis objects.

    All synthesis objects share a variety of similarities and thus need
    to have similar methods. Some of these can be implemented here and
    simply inherited, some of them will need to be different for each
    sub-class and thus are marked as abstract methods here

    """

    @abc.abstractmethod
    def synthesize(self):
        r"""Synthesize something."""
        pass

    def save(self, file_path: str, attrs: list[str] | None = None):
        r"""Save all relevant (non-model) variables in .pt file.

        If you leave attrs as None, we grab vars(self) and exclude 'model'.
        This is probably correct, but the option is provided to override it
        just in case

        Parameters
        ----------
        file_path : str
            The path to save the synthesis object to
        attrs : list or None, optional
            List of strs containing the names of the attributes of this
            object to save. See above for behavior if attrs is None.

        """
        if attrs is None:
            # this copies the attributes dict so we don't actually remove the
            # model attribute in the next line
            attrs = {k: v for k, v in vars(self).items()}
            attrs.pop("_model", None)

        save_dict = {}
        for k in attrs:
            if k == "_model":
                warnings.warn(
                    "Models can be quite large and they don't change"
                    " over synthesis. Please be sure that you "
                    "actually want to save the model."
                )
            attr = getattr(self, k)
            # detaching the tensors avoids some headaches like the
            # tensors having extra hooks or the like
            if isinstance(attr, torch.Tensor):
                attr = attr.detach()
            save_dict[k] = attr
        torch.save(save_dict, file_path)

    def load(
        self,
        file_path: str,
        map_location: str | None = None,
        check_attributes: list[str] = [],
        check_loss_functions: list[str] = [],
        **pickle_load_args,
    ):
        r"""Load all relevant attributes from a .pt file.

        This should be called by an initialized ``Synthesis`` object -- we will
        ensure that the attributes in the ``check_attributes`` arg all match in
        the current and loaded object.

        Note this operates in place and so doesn't return anything.

        Parameters
        ----------
        file_path :
            The path to load the synthesis object from
        map_location :
            map_location argument to pass to ``torch.load``. If you save
            stuff that was being run on a GPU and are loading onto a
            CPU, you'll need this to make sure everything lines up
            properly. This should be structured like the str you would
            pass to ``torch.device``
        check_attributes :
            List of strings we ensure are identical in the current
            ``Synthesis`` object and the loaded one. Checking the model is
            generally not recommended, since it can be hard to do (checking
            callable objects is hard in Python) -- instead, checking the
            ``base_representation`` should ensure the model hasn't functinoally
            changed.
        check_loss_functions :
            Names of attributes that are loss functions and so must be checked
            specially -- loss functions are callables, and it's very difficult
            to check python callables for equality so, to get around that, we
            instead call the two versions on the same pair of tensors,
            and compare the outputs.

        pickle_load_args :
            any additional kwargs will be added to ``pickle_module.load`` via
            ``torch.load``, see that function's docstring for details.

        """
        tmp_dict = torch.load(file_path, map_location=map_location, **pickle_load_args)
        if map_location is not None:
            device = map_location
        else:
            for v in tmp_dict.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    break
        for k in check_attributes:
            # The only hidden attributes we'd check are those like
            # range_penalty_lambda, where this function is checking the
            # hidden version (which starts with '_'), but during
            # initialization, the user specifies the version without
            # the initial underscore. This is because this function
            # needs to be able to set the attribute, which can only be
            # done with the hidden version.
            display_k = k[1:] if k.startswith("_") else k
            if not hasattr(self, k):
                raise AttributeError(
                    "All values of `check_attributes` should be "
                    "attributes set at initialization, but got "
                    f"attr {display_k}!"
                )
            if isinstance(getattr(self, k), torch.Tensor):
                # there are two ways this can fail -- the first is if they're
                # the same shape but different values and the second (in the
                # except block) are if they're different shapes.
                try:
                    if not torch.allclose(
                        getattr(self, k).to(tmp_dict[k].device),
                        tmp_dict[k],
                        rtol=5e-2,
                    ):
                        raise ValueError(
                            f"Saved and initialized {display_k} are "
                            f"different! Initialized: {getattr(self, k)}"
                            f", Saved: {tmp_dict[k]}, difference: "
                            f"{getattr(self, k) - tmp_dict[k]}"
                        )
                except RuntimeError as e:
                    # we end up here if dtype or shape don't match
                    if "The size of tensor a" in e.args[0]:
                        raise RuntimeError(
                            f"Attribute {display_k} have different shapes in"
                            " saved and initialized versions! Initialized"
                            f": {getattr(self, k).shape}, Saved: "
                            f"{tmp_dict[k].shape}"
                        )
                    elif "did not match" in e.args[0]:
                        raise RuntimeError(
                            f"Attribute {display_k} has different dtype in "
                            "saved and initialized versions! Initialized"
                            f": {getattr(self, k).dtype}, Saved: "
                            f"{tmp_dict[k].dtype}"
                        )
                    else:
                        raise e
            elif isinstance(getattr(self, k), float):
                if not np.allclose(getattr(self, k), tmp_dict[k]):
                    raise ValueError(
                        f"Saved and initialized {display_k} are different!"
                        f" Self: {getattr(self, k)}, "
                        f"Saved: {tmp_dict[k]}"
                    )
            else:
                if getattr(self, k) != tmp_dict[k]:
                    raise ValueError(
                        f"Saved and initialized {display_k} are different!"
                        f" Self: {getattr(self, k)}, "
                        f"Saved: {tmp_dict[k]}"
                    )
        for k in check_loss_functions:
            # same as above
            display_k = k[1:] if k.startswith("_") else k
            # this way, we know it's the right shape
            tensor_a, tensor_b = torch.rand(2, *self._image_shape).to(device)
            saved_loss = tmp_dict[k](tensor_a, tensor_b)
            init_loss = getattr(self, k)(tensor_a, tensor_b)
            if not torch.allclose(saved_loss, init_loss, rtol=1e-2):
                raise ValueError(
                    f"Saved and initialized {display_k} are "
                    "different! On two random tensors: "
                    f"Initialized: {init_loss}, Saved: "
                    f"{saved_loss}, difference: "
                    f"{init_loss - saved_loss}"
                )
        for k, v in tmp_dict.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def to(self, *args, attrs: list[str] = [], **kwargs):
        r"""Moves and/or casts the parameters and buffers.
        Similar to ``save``, this is an abstract method only because you
        need to define the attributes to call to on.

        This can be called as
        .. function:: to(device=None, dtype=None, non_blocking=False)
        .. function:: to(dtype, non_blocking=False)
        .. function:: to(tensor, non_blocking=False)
        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices. When calling this method to move tensors
        to a CUDA device, items in ``attrs`` that start with "saved_" will not
        be moved.
        .. note::
            This method modifies the module in-place.
        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            attrs (:class:`list`): list of strs containing the attributes of
                this object to move to the specified device/dtype
        """
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        def move(a, k):
            move_device = None if k.startswith("saved_") else device
            if memory_format is not None and a.dim() == 4:
                return a.to(
                    move_device,
                    dtype,
                    non_blocking,
                    memory_format=memory_format,
                )
            else:
                return a.to(move_device, dtype, non_blocking)

        for k in attrs:
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(attr, torch.Tensor):
                    attr = move(attr.data, k)
                    if isinstance(getattr(self, k), torch.nn.Parameter):
                        attr = torch.nn.Parameter(attr)
                    if getattr(self, k).requires_grad:
                        attr = attr.requires_grad_()
                    setattr(self, k, attr)
                elif isinstance(attr, list):
                    setattr(self, k, [move(a, k) for a in attr])
                elif attr is not None:
                    setattr(self, k, move(attr, k))


class OptimizedSynthesis(Synthesis):
    r"""Abstract super-class for synthesis objects that use optimization.

    The primary difference between this and the generic Synthesis class is that
    these will use an optimizer object to iteratively update their output.

    """

    def __init__(
        self,
        range_penalty_lambda: float = 0.1,
        allowed_range: tuple[float, float] = (0, 1),
    ):
        """Initialize the properties of OptimizedSynthesis."""
        self._losses = []
        self._gradient_norm = []
        self._pixel_change_norm = []
        self._store_progress = None
        self._optimizer = None
        if range_penalty_lambda < 0:
            raise Exception("range_penalty_lambda must be non-negative!")
        self._range_penalty_lambda = range_penalty_lambda
        self._allowed_range = allowed_range

    @abc.abstractmethod
    def _initialize(self):
        r"""What to start synthesis with."""
        pass

    @abc.abstractmethod
    def objective_function(self):
        r"""How good is the current synthesized object.

        See ``plenoptic.tools.optim`` for some examples.
        """
        pass

    @abc.abstractmethod
    def _check_convergence(self):
        r"""How to determine if synthesis has finished.

        See ``plenoptic.tools.convergence`` for some examples.
        """
        pass

    def _closure(self) -> torch.Tensor:
        r"""An abstraction of the gradient calculation, before the optimization step.

        This enables optimization algorithms that perform several evaluations
        of the gradient before taking a step (ie. second order methods like
        LBFGS).

        Additionally, this is where ``loss`` is calculated and
        ``loss.backward()`` is called.

        Returns
        -------
        loss
            Loss of the current objective function

        """
        self.optimizer.zero_grad()
        loss = self.objective_function()
        loss.backward(retain_graph=False)
        return loss

    def _initialize_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None,
        synth_name: str,
        learning_rate: float = 0.01,
    ):
        """Initialize optimizer.

        First time this is called, optimizer can be:

        - None, in which case we create an Adam optimizer with amsgrad=True and
          ``lr=learning_rate`` with a single parameter, the synthesis attribute

        - torch.optim.Optimizer, in which case it must already have the
          synthesis attribute (e.g., metamer) as its only parameter.

        The synthesis attribute is the one with the name ``synth_name``

        Every subsequent time (so, when resuming synthesis), optimizer must be
        None (and we use the original optimizer object).

        """
        synth_attr = getattr(self, synth_name)
        if optimizer is None:
            if self.optimizer is None:
                self._optimizer = torch.optim.Adam(
                    [synth_attr], lr=learning_rate, amsgrad=True
                )
        else:
            if self.optimizer is not None:
                raise TypeError("When resuming synthesis, optimizer arg must be None!")
            params = optimizer.param_groups[0]["params"]
            if len(params) != 1 or not torch.equal(params[0], synth_attr):
                raise ValueError(
                    f"For {synth_name} synthesis, optimizer must have one "
                    f"parameter, the {synth_name} we're synthesizing."
                )
            self._optimizer = optimizer

    @property
    def range_penalty_lambda(self):
        return self._range_penalty_lambda

    @property
    def allowed_range(self):
        return self._allowed_range

    @property
    def losses(self):
        """Synthesis loss over iterations."""
        return torch.as_tensor(self._losses)

    @property
    def gradient_norm(self):
        """Synthesis gradient's L2 norm over iterations."""
        return torch.as_tensor(self._gradient_norm)

    @property
    def pixel_change_norm(self):
        """L2 norm change in pixel values over iterations."""
        return torch.as_tensor(self._pixel_change_norm)

    @property
    def store_progress(self):
        return self._store_progress

    @store_progress.setter
    def store_progress(self, store_progress: bool | int):
        """Initialize store_progress.

        Sets the ``self.store_progress`` attribute, as well as changing the
        ``saved_metamer`` attibute to a list so we can append to them. finally,
        adds first value to ``saved_metamer`` if it's empty.

        Parameters
        ----------
        store_progress : bool or int, optional
            Whether we should store the metamer image in progress on every
            iteration. If False, we don't save anything. If True, we save every
            iteration. If an int, we save every ``store_progress`` iterations
            (note then that 0 is the same as False and 1 the same as True). If
            True or int>0, ``self.saved_metamer`` contains the stored images.

        """
        if store_progress and store_progress is True:
            store_progress = 1
        if self.store_progress is not None and store_progress != self.store_progress:
            # we require store_progress to be the same because otherwise the
            # subsampling relationship between attrs that are stored every
            # iteration (loss, gradient, etc) and those that are stored every
            # store_progress iteration (e.g., saved_metamer) changes partway
            # through and that's annoying
            raise Exception(
                "If you've already run synthesize() before, must "
                "re-run it with same store_progress arg. You "
                f"passed {store_progress} instead of "
                f"{self.store_progress} (True is equivalent to 1)"
            )
        self._store_progress = store_progress

    @property
    def optimizer(self):
        return self._optimizer
