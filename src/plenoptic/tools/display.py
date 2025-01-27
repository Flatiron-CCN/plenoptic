"""various helpful utilities for plotting or displaying information"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyrtools as pt
import torch

from .data import to_numpy


def imshow(
    image,
    vrange="indep1",
    zoom=None,
    title="",
    col_wrap=None,
    ax=None,
    cmap=None,
    plot_complex="rectangular",
    batch_idx=None,
    channel_idx=None,
    as_rgb=False,
    **kwargs,
):
    """Show image(s) correctly.

    This function shows images correctly, making sure that each element in the
    tensor corresponds to a pixel or an integer number of pixels, to avoid
    aliasing (NOTE: this guarantee only holds for the saved image; it should
    generally hold in notebooks as well, but will fail if, e.g., you plot an
    image that's 2000 pixels wide on an monitor 1000 pixels wide; the notebook
    handles the rescaling in a way we can't control).

    Parameters
    ----------
    image : torch.Tensor or list
        The images to display. Tensors should be 4d (batch, channel, height,
        width). List of tensors should be used for tensors of different height
        and width: all images will automatically be rescaled so they're
        displayed at the same height and width, thus, their heights and widths
        must be scalar multiples of each other.
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to
        the minimum and maximum value of the colormap, respectively. If a
        string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute
                     value, and come from the minimum or maximum across all
                     images, whichever has the larger absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the
                          minimum/maximum values across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across
                     all images) minus/ plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the
                     10th/90th percentile values to the 10th/90th percentile of
                     the display intensity range. For example: vmin is the 10th
                     percentile image value minus 1/8 times the difference
                     between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the
                      same absolute value, which comes from either their
                      minimum or maximum value, whichever has the larger
                      absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their
                      minimum/maximum values
        * `'indep2'`: each image has an independent vmin/vmax, which is their
                      mean minus/plus 2 std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that
                      the 10th/90th percentile values map to the 10th/90th
                      percentile intensities.
    zoom : `float` or `None`
        ratio of display pixels to image pixels. if >1, must be an integer. If
        <1, must be 1/d where d is a a divisor of the size of the largest
        image. If None, we try to determine the best zoom.
    title : `str`, `list`, or None, optional
        Title for the plot. In addition to the specified title, we add a
        subtitle giving the plotted range and dimensionality (with zoom)
        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img,
          assigning each title to corresponding image.
        * if None, no title will be printed (and subtitle will be removed).
    col_wrap : `int` or None, optional
        number of axes to have in each row. If None, will fit all axes in a
        single row.
    ax : `matplotlib.pyplot.axis` or None, optional
        if None, we make the appropriate figure. otherwise, we resize the axes
        so that it's the appropriate number of pixels (done by shrinking the
        bbox - if the bbox is already too small, this will throw an Exception!,
        so first define a large enough figure using either make_figure or
        plt.figure)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
        for any other value, we raise a warning and default to rectangular.
    batch_idx : int or None, optional
        Which element from the batch dimension to plot. If None, we plot all.
    channel_idx : int or None, optional
        Which element from the channel dimension to plot. If None, we plot all.
        Note if this is an int, then `as_rgb=True` will fail, because we
        restrict the channels.
    as_rgb : bool, optional
        Whether to consider the channels as encoding RGB(A) values. If True, we
        attempt to plot the image in color, so your tensor must have 3 (or 4 if
        you want the alpha channel) elements in the channel dimension, or this
        will raise an Exception. If False, we plot each channel as a separate
        grayscale image.
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    fig : `PyrFigure`
        figure containing the plotted images

    """
    if not isinstance(image, list):
        image = [image]
    images_to_plot = []
    heights, widths = [], []
    for im in image:
        im = to_numpy(im)
        if im.shape[0] > 1 and batch_idx is not None:
            # this preserves the number of dimensions
            im = im[batch_idx : batch_idx + 1]
        if channel_idx is not None:
            # this preserves the number of dimensions
            im = im[:, channel_idx : channel_idx + 1]
        # allow RGB and RGBA
        if as_rgb:
            if im.shape[1] not in [3, 4]:
                raise Exception(
                    "If as_rgb is True, then channel must have 3 or 4 elements!"
                )
            im = im.transpose(0, 2, 3, 1)
            # want to insert a fake "channel" dimension here, so our putting it
            # into a list below works as expected
            im = im.reshape((im.shape[0], 1, *im.shape[1:]))
        elif im.shape[1] > 1 and im.shape[0] > 1:
            raise Exception(
                "Don't know how to plot images with more than one channel and"
                " batch! Use batch_idx / channel_idx to choose a subset for"
                " plotting"
            )
        # by iterating through it twice, we make sure to peel apart the batch
        # and channel dimensions so that they each show up as a separate image.
        # because of how we've handled everything above, we know that im will
        # be (b,c,h,w) or (b,c,h,w,r) where r is the RGB(A) values
        for i in im:
            # at this point, i_ are all shape (h,w) or (h,w,r) and so we don't
            # squeeze, which could accidentally drop a dimension if h or w is a
            # singleton dimension
            images_to_plot.extend([i_ for i_ in i])
            heights.extend([i_.shape[0] for i_ in i])
            widths.extend([i_.shape[1] for i_ in i])

    def find_zoom(x, limit):
        """Find zoom that works. This is only for limit < x."""
        # find all non-trivial divisors of x
        divisors = [i for i in range(2, x) if not x % i]
        # find the largest zoom (equivalently, smallest divisor) such that the
        # zoomed in image is smaller than the limit
        return 1 / min([i for i in divisors if x / i <= limit])

    if ax is not None and zoom is None:
        if ax.bbox.height > max(heights):
            zoom = ax.bbox.height // max(heights)
        else:
            zoom = find_zoom(max(heights), ax.bbox.height)
        if ax.bbox.width > max(widths):
            zoom = min(zoom, ax.bbox.width // max(widths))
        else:
            zoom = find_zoom(max(widths), ax.bbox.width)
    elif zoom is None:
        zoom = 1
    return pt.imshow(
        images_to_plot,
        vrange=vrange,
        zoom=zoom,
        title=title,
        col_wrap=col_wrap,
        ax=ax,
        cmap=cmap,
        plot_complex=plot_complex,
        **kwargs,
    )


def animshow(
    video,
    framerate=2.0,
    repeat=False,
    vrange="indep1",
    zoom=1,
    title="",
    col_wrap=None,
    ax=None,
    cmap=None,
    plot_complex="rectangular",
    batch_idx=None,
    channel_idx=None,
    as_rgb=False,
    **kwargs,
):
    """Animate video(s) correctly.

    This function animates videos correctly, making sure that each element in
    the tensor corresponds to a pixel or an integer number of pixels, to avoid
    aliasing (NOTE: this guarantee only holds for the saved animation (assuming
    video compression doesn't interfere); it should generally hold in notebooks
    as well, but will fail if, e.g., your video is 2000 pixels wide on an
    monitor 1000 pixels wide; the notebook handles the rescaling in a way we
    can't control).

    This functions returns a matplotlib FuncAnimation object. See our documentation
    (e.g.,
    [Quickstart](https://docs.plenoptic.org/docs/branch/main/tutorials/00_quickstart.html))
    for examples on how to view it in a Jupyter notebook. In order to save, use
    ``anim.save(filename)``. In either case, this can take a while and you'll need the
    appropriate writer installed and on your path, e.g., ffmpeg, imagemagick, etc). See
    [matplotlib documentation](https://matplotlib.org/stable/api/animation_api.html) for
    more details.

    Parameters
    ----------
    video : torch.Tensor or list
        The videos to display. Tensors should be 5d (batch, channel, time,
        height, width). List of tensors should be used for tensors of different
        height and width: all videos will automatically be rescaled so they're
        displayed at the same height and width, thus, their heights and widths
        must be scalar multiples of each other. Videos must all have the same
        number of frames as well.
    framerate : `float`
        Temporal resolution of the video, in Hz (frames per second).
    repeat : `bool`
        whether to loop the animation or just play it once
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to
        the minimum and maximum value of the colormap, respectively. If a
        string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute
                     value, and come from the minimum or maximum across all
                     images, whichever has the larger absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the
                          minimum/maximum values across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across
                     all images) minus/ plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the
                     10th/90th percentile values to the 10th/90th percentile of
                     the display intensity range. For example: vmin is the 10th
                     percentile image value minus 1/8 times the difference
                     between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the
                      same absolute value, which comes from either their
                      minimum or maximum value, whichever has the larger
                      absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their
                      minimum/maximum values
        * `'indep2'`: each image has an independent vmin/vmax, which is their
                      mean minus/plus 2 std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that
                      the 10th/90th percentile values map to the 10th/90th
                      percentile intensities.
    zoom : `float`
        ratio of display pixels to image pixels. if >1, must be an integer. If
        <1, must be 1/d where d is a a divisor of the size of the largest
        image.
    title : `str`, `list`, or None, optional
        Title for the plot. In addition to the specified title, we add a
        subtitle giving the plotted range and dimensionality (with zoom)
        * if `str`, will put the same title on every plot.
        * if `list`, all values must be `str`, must be the same length as img,
          assigning each title to corresponding image.
        * if None, no title will be printed (and subtitle will be removed).
    col_wrap : `int` or None, optional
        number of axes to have in each row. If None, will fit all axes in a
        single row.
    ax : `matplotlib.pyplot.axis` or None, optional
        if None, we make the appropriate figure. otherwise, we resize the axes
        so that it's the appropriate number of pixels (done by shrinking the
        bbox - if the bbox is already too small, this will throw an Exception!,
        so first define a large enough figure using either
        pyrtools.make_figure or plt.figure)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
        for any other value, we raise a warning and default to rectangular.
    batch_idx : int or None, optional
        Which element from the batch dimension to plot. If None, we plot all.
    channel_idx : int or None, optional
        Which element from the channel dimension to plot. If None, we plot all.
        Note if this is an int, then `as_rgb=True` will fail, because we
        restrict the channels.
    as_rgb : bool, optional
        Whether to consider the channels as encoding RGB(A) values. If True, we
        attempt to plot the image in color, so your tensor must have 3 (or 4 if
        you want the alpha channel) elements in the channel dimension, or this
        will raise an Exception. If False, we plot each channel as a separate
        grayscale image.
    kwargs :
        Passed to `ax.imshow`

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. In order to view, must convert to HTML
        or save.

    Notes
    -----

    By default, we use the ffmpeg backend, which requires that you have
    ffmpeg installed and on your path (https://ffmpeg.org/download.html).
    To use a different, use the matplotlib rcParams:
    `matplotlib.rcParams['animation.writer'] = writer`, see
    https://matplotlib.org/stable/api/animation_api.html#writer-classes for
    more details.

    """
    if not isinstance(video, list):
        video = [video]
    videos_to_show = []
    for vid in video:
        vid = to_numpy(vid)
        if vid.shape[0] > 1 and batch_idx is not None:
            # this preserves the number of dimensions
            vid = vid[batch_idx : batch_idx + 1]
        if channel_idx is not None:
            # this preserves the number of dimensions
            vid = vid[:, channel_idx : channel_idx + 1]
        # allow RGB and RGBA
        if as_rgb:
            if vid.shape[1] not in [3, 4]:
                raise Exception(
                    "If as_rgb is True, then channel must have 3 or 4 elements!"
                )
            vid = vid.transpose(0, 2, 3, 4, 1)
            # want to insert a fake "channel" dimension here, so our putting it
            # into a list below works as expected
            vid = vid.reshape((vid.shape[0], 1, *vid.shape[1:]))
        elif vid.shape[1] > 1 and vid.shape[0] > 1:
            raise Exception(
                "Don't know how to plot images with more than one channel and"
                " batch! Use batch_idx / channel_idx to choose a subset for"
                " plotting"
            )
        # by iterating through it twice, we make sure to peel apart the batch
        # and channel dimensions so that they each show up as a separate video.
        # because of how we've handled everything above, we know that vid will
        # be (b,c,t,h,w) or (b,c,t,h,w,r) where r is the RGB(A) values
        for v in vid:
            videos_to_show.extend([v_.squeeze() for v_ in v])
    return pt.animshow(
        videos_to_show,
        framerate=framerate,
        as_html5=False,
        repeat=repeat,
        vrange=vrange,
        zoom=zoom,
        title=title,
        col_wrap=col_wrap,
        ax=ax,
        cmap=cmap,
        plot_complex=plot_complex,
        **kwargs,
    )


def pyrshow(
    pyr_coeffs,
    vrange="indep1",
    zoom=1,
    show_residuals=True,
    cmap=None,
    plot_complex="rectangular",
    batch_idx=0,
    channel_idx=0,
    **kwargs,
):
    r"""Display steerable pyramid coefficients in orderly fashion.

    This function uses ``imshow`` to show the coefficients of the steeable
    pyramid, such that each scale shows up on a single row, with each scale in
    a given column.

    Note that unlike imshow, we can only show one batch or channel at a time

    Parameters
    ----------
    pyr_coeffs : `dict`
        pyramid coefficients in the standard dictionary format as returned by
        ``SteerablePyramidFreq.forward()``
    vrange : `tuple` or `str`
        If a 2-tuple, specifies the image values vmin/vmax that are mapped to
        the minimum and maximum value of the colormap, respectively. If a
        string:

        * `'auto0'`: all images have same vmin/vmax, which have the same absolute
                     value, and come from the minimum or maximum across all
                     images, whichever has the larger absolute value
        * `'auto/auto1'`: all images have same vmin/vmax, which are the
                          minimum/maximum values across all images
        * `'auto2'`: all images have same vmin/vmax, which are the mean (across
                     all images) minus/ plus 2 std dev (across all images)
        * `'auto3'`: all images have same vmin/vmax, chosen so as to map the
                     10th/90th percentile values to the 10th/90th percentile of
                     the display intensity range. For example: vmin is the 10th
                     percentile image value minus 1/8 times the difference
                     between the 90th and 10th percentile
        * `'indep0'`: each image has an independent vmin/vmax, which have the
                      same absolute value, which comes from either their
                      minimum or maximum value, whichever has the larger
                      absolute value.
        * `'indep1'`: each image has an independent vmin/vmax, which are their
                      minimum/maximum values
        * `'indep2'`: each image has an independent vmin/vmax, which is their
                      mean minus/plus 2 std dev
        * `'indep3'`: each image has an independent vmin/vmax, chosen so that
                      the 10th/90th percentile values map to the 10th/90th
                      percentile intensities.
    zoom : `float`
        ratio of display pixels to image pixels. if >1, must be an integer. If
        <1, must be 1/d where d is a a divisor of the size of the largest
        image.
    show_residuals : `bool`
        whether to display the residual bands (lowpass, highpass depending on the
        pyramid type)
    cmap : matplotlib colormap, optional
        colormap to use when showing these images
    plot_complex : {'rectangular', 'polar', 'logpolar'}
        specifies handling of complex values.

        * `'rectangular'`: plot real and imaginary components as separate images
        * `'polar'`: plot amplitude and phase as separate images
        * `'logpolar'`: plot log_2 amplitude and phase as separate images
        for any other value, we raise a warning and default to rectangular.
    batch_idx : int, optional
        Which element from the batch dimension to plot.
    channel_idx : int, optional
        Which element from the channel dimension to plot.
    kwargs :
        Passed on to ``pyrtools.pyrshow``

    Returns
    -------
    fig: `PyrFigure`
        the figure displaying the coefficients.

    """
    pyr_coeffvis = {}
    is_complex = False
    for k, v in pyr_coeffs.items():
        im = to_numpy(v)
        if np.iscomplex(im).any():
            is_complex = True
        # this removes only the first (batch) dimension
        im = im[batch_idx : batch_idx + 1].squeeze(0)
        # this removes only the first (now channel) dimension
        im = im[channel_idx : channel_idx + 1].squeeze(0)
        # because of how we've handled everything above, we know that im will
        # be (h,w).
        pyr_coeffvis[k] = im

    return pt.pyrshow(
        pyr_coeffvis,
        is_complex=is_complex,
        vrange=vrange,
        zoom=zoom,
        cmap=cmap,
        plot_complex=plot_complex,
        show_residuals=show_residuals,
        **kwargs,
    )


def clean_up_axes(
    ax,
    ylim=None,
    spines_to_remove=["top", "right", "bottom"],
    axes_to_remove=["x"],
):
    r"""Clean up an axis, as desired when making a stem plot of the representation

    Parameters
    ----------
    ax : `matplotlib.pyplot.axis`
        The axis to clean up.
    ylim : `tuple`, False, or None
        If a tuple, the y-limits to use for this plot. If None, we use the
        default, slightly adjusted so that the minimum is 0. If False,
        we do nothing.
    spines_to_remove : `list`
        Some combination of 'top', 'right', 'bottom', and 'left'. The spines we
        remove from the axis.
    axes_to_remove : `list`
        Some combination of 'x', 'y'. The axes to set as invisible.

    Returns
    -------
    ax : matplotlib.pyplot.axis
        The cleaned-up axis

    """
    if spines_to_remove is None:
        spines_to_remove = ["top", "right", "bottom"]
    if axes_to_remove is None:
        axes_to_remove = ["x"]

    if ylim is not None:
        if ylim:
            ax.set_ylim(ylim)
    else:
        ax.set_ylim((0, ax.get_ylim()[1]))
    if "x" in axes_to_remove:
        ax.xaxis.set_visible(False)
    if "y" in axes_to_remove:
        ax.yaxis.set_visible(False)
    for s in spines_to_remove:
        ax.spines[s].set_visible(False)
    return ax


def update_stem(stem_container, ydata):
    r"""Update the information in a stem plot

    We update the information in a single stem plot to match that given
    by ``ydata``. We update the position of the markers and and the
    lines connecting them to the baseline, but we don't change the
    baseline at all and assume that the xdata shouldn't change at all.

    Parameters
    ----------
    stem_container : `matplotlib.container.StemContainer`
        Single container for the artists created in a ``plt.stem``
        plot. It can be treated like a namedtuple ``(markerline,
        stemlines, baseline)``. In order to get this from an axis
        ``ax``, try ``ax.containers[0]`` (obviously if you have more
        than one container in that axis, it may not be the first one).
    ydata : array_like
        The new y-data to show on the plot. Importantly, must be the
        same length as the existing y-data.

    Returns
    -------
    stem_container : `matplotlib.container.StemContainer`
        The StemContainer containing the updated artists.

    """
    stem_container.markerline.set_ydata(ydata)
    segments = stem_container.stemlines.get_segments().copy()
    for s, y in zip(segments, ydata):
        try:
            s[1, 1] = y
        except IndexError:
            # this happens when our segment array is 1x2 instead of 2x2,
            # which is the case when the data there is nan
            continue
    stem_container.stemlines.set_segments(segments)
    return stem_container


def rescale_ylim(axes, data):
    r"""rescale y-limits nicely

    We take the axes and set their limits to be ``(-y_max, y_max)``,
    where ``y_max=np.abs(data).max()``

    Parameters
    ----------
    axes : `list`
        A list of matplotlib axes to rescale
    data : array_like or dict
        The data to use when rescaling (or a dictiontary of those
        values)
    """
    data = data.cpu()

    def find_ymax(data):
        try:
            return np.abs(data).max()
        except RuntimeError:
            # then we need to call to_numpy on it because it needs to be
            # detached and converted to an array
            return np.abs(to_numpy(data)).max()

    try:
        y_max = find_ymax(data)
    except TypeError:
        # then this is a dictionary
        y_max = np.max([find_ymax(d) for d in data.values()])
    for ax in axes:
        ax.set_ylim((-y_max, y_max))


def clean_stem_plot(data, ax=None, title="", ylim=None, xvals=None, **kwargs):
    r"""convenience wrapper for plotting stem plots

    This plots the data, baseline, cleans up the axis, and sets the
    title

    Should not be called by users directly, but is a helper function for
    the various plot_representation() functions

    By default, stem plot would have a baseline that covers the entire
    range of the data. We want to be able to break that up visually (so
    there's a line from 0 to 9, from 10 to 19, etc), and passing xvals
    separately allows us to do that. If you want the default stem plot
    behavior, leave xvals as None.

    Parameters
    ----------
    data : `np.ndarray`
        The data to plot (as a stem plot)
    ax : `matplotlib.pyplot.axis` or `None`, optional
        The axis to plot the data on. If None, we plot on the current
        axis
    title : str or None, optional
        The title to put on the axis if not None. If None, we don't call
        ``ax.set_title`` (useful if you want to avoid changing the title
        on an existing plot)
    ylim : tuple or None, optional
        If not None, the y-limits to use for this plot. If None, we use the
        default, slightly adjusted so that the minimum is 0. If False, do not
        change y-limits.
    xvals : `tuple` or `None`, optional
        A 2-tuple of lists, containing the start (``xvals[0]``) and stop
        (``xvals[1]``) x values for plotting. If None, we use the
        default stem plot behavior.
    kwargs :
        passed to ax.stem

    Returns
    -------
    ax : `matplotlib.pyplot.axis`
        The axis with the plot

    Examples
    --------
    We allow for breaks in the baseline value if we want to visually
    break up the plot, as we see below.

    ..plot::
      :include-source:

      import plenoptic as po
      import numpy as np
      import matplotlib.pyplot as plt
      # if ylim=None, as in this example, the minimum y-valuewill get
      # set to 0, so we want to make sure our values are all positive
      y = np.abs(np.random.randn(55))
      y[15:20] = np.nan
      y[35:40] = np.nan
      # we want to draw the baseline from 0 to 14, 20 to 34, and 40 to
      # 54, everywhere that we have non-NaN values for y
      xvals = ([0, 20, 40], [14, 34, 54])
      po.tools.display.clean_stem_plot(y,  xvals=xvals)
      plt.show()

    If we don't care about breaking up the x-axis, you can simply use
    the default xvals (``None``). In this case, this function will just
    clean up the plot a little bit

    ..plot::
      :include-source:

      import plenoptic as po
      import numpy as np
      import matplotlib.pyplot as plt
      # if ylim=None, as in this example, the minimum y-valuewill get
      # set to 0, so we want to make sure our values are all positive
      y = np.abs(np.random.randn(55))
      po.tools.display.clean_stem_plot(y)
      plt.show()

    """
    if ax is None:
        ax = plt.gca()
    if xvals is not None:
        basefmt = " "
        ax.hlines(len(xvals[0]) * [0], xvals[0], xvals[1], colors="C3", zorder=10)
    else:
        # this is the default basefmt value
        basefmt = None
    ax.stem(data, basefmt=basefmt, **kwargs)
    ax = clean_up_axes(ax, ylim, ["top", "right", "bottom"])
    if title is not None:
        ax.set_title(title)
    return ax


def _get_artists_from_axes(axes, data):
    """Grab artists from axes.

    For now, we only grab containers (stem plots), images, or lines

    See the docstring of :meth:`update_plot()` for details on how `axes` and
    `data` should be structured

    Parameters
    ----------
    axes : list or matplotlib.axes.Axes
        The axis/axes to update.
    data : torch.Tensor or dict
        The new data to plot.

    Returns
    -------
    artists : dict
        dictionary of artists for updating plots. values are the artists to
        use, keys are the corresponding keys for data

    """
    if not hasattr(axes, "__iter__"):
        # then we only have one axis, so we may be able to update more than one
        # data element.
        if len(axes.containers) > 0:
            data_check = 1
            artists = axes.containers
        elif len(axes.images) > 0:
            # images are weird, so don't check them like this
            data_check = None
            artists = axes.images
        elif len(axes.lines) > 0:
            data_check = 1
            artists = axes.lines
        elif len(axes.collections) > 0:
            data_check = 2
            artists = axes.collections
        if isinstance(data, dict):
            artists = {ax.get_label(): ax for ax in artists}
        else:
            if data_check == 1 and data.shape[1] != len(artists):
                raise Exception(
                    f"data has {data.shape[1]} things to plot, but "
                    f"your axis contains {len(artists)} plotting artists, "
                    "so unsure how to continue! Pass data as a dictionary"
                    " with keys corresponding to the labels of the artists"
                    " to update to resolve this."
                )
            elif data_check == 2 and data.ndim > 2 and data.shape[-3] != len(artists):
                raise Exception(
                    f"data has {data.shape[-3]} things to plot, but "
                    f"your axis contains {len(artists)} plotting artists, "
                    "so unsure how to continue! Pass data as a dictionary"
                    " with keys corresponding to the labels of the artists"
                    " to update to resolve this."
                )
    else:
        # then we have multiple axes, so we are only updating one data element
        # per plot
        artists = []
        for ax in axes:
            if len(ax.containers) == 1:
                data_check = 1
                artists.extend(ax.containers)
            elif len(ax.images) == 1:
                # images are weird, so don't check them like this
                data_check = None
                artists.extend(ax.images)
            elif len(ax.lines) == 1:
                artists.extend(ax.lines)
                data_check = 1
            elif len(ax.collections) == 1:
                artists.extend(ax.collections)
                data_check = 2
        if isinstance(data, dict):
            if len(data.keys()) != len(artists):
                raise Exception(
                    f"data has {len(data.keys())} things to plot, but "
                    f"you passed {len(axes)} axes , so unsure how "
                    "to continue!"
                )
            artists = {k: a for k, a in zip(data.keys(), artists)}
        else:
            if data_check == 1 and data.shape[1] != len(artists):
                raise Exception(
                    f"data has {data.shape[1]} things to plot, but "
                    f"you passed {len(axes)} axes , so unsure how "
                    "to continue!"
                )
            if data_check == 2 and data.ndim > 2 and data.shape[-3] != len(artists):
                raise Exception(
                    f"data has {data.shape[-3]} things to plot, but "
                    f"you passed {len(axes)} axes , so unsure how "
                    "to continue!"
                )
    if not isinstance(artists, dict):
        artists = {f"{i:02d}": a for i, a in enumerate(artists)}
    return artists


def update_plot(axes, data, model=None, batch_idx=0):
    r"""Update the information in some axes.

    This is used for creating an animation over time. In order to create
    the animation, we need to know how to update the matplotlib Artists,
    and this provides a simple way of doing that. It assumes the plot
    has been created by something like ``plot_representation``, which
    initializes all the artists.

    We can update stem plots, lines (as returned by ``plt.plot``), scatter
    plots, or images (RGB, RGBA, or grayscale).

    There are two modes for this:

    - single axis: axes is a single axis, which may contain multiple artists
      (all of the same type) to update. data should be a Tensor with multiple
      channels (one per artist in the same order) or be a dictionary whose keys
      give the label(s) of the corresponding artist(s) and whose values are
      Tensors.

    - multiple axes: axes is a list of axes, each of which contains a single
      artist to update (artists can be different types). data should be a
      Tensor with multiple channels (one per axis in the same order) or a
      dictionary with the same number of keys as axes, which we can iterate
      through in order, and whose values are Tensors.

    In all cases, data Tensors should be 3d (if the plot we're updating is a
    line or stem plot) or 4d (if it's an image or scatter plot).

    RGB(A) images are special, since we store that info along the channel
    dimension, so they only work with single-axis mode (which will only have a
    single artist, because that's how imshow works).

    If you have multiple axes, each with multiple artists you want to update,
    that's too complicated for us, and so you should write a
    ``model.update_plot()`` function which handles that.

    If ``model`` is set, we try to call ``model.update_plot()`` (which
    must also return artists). If model doesn't have an ``update_plot``
    method, then we try to figure out how to update the axes ourselves,
    based on the shape of the data.

    Parameters
    ----------
    axes : `list` or `matplotlib.pyplot.axis`
        The axis or list of axes to update. We assume that these are the axes
        created by ``plot_representation`` and so contain stem plots in the
        correct order.
    data : `torch.Tensor` or `dict`
        The new data to plot.
    model : `torch.nn.Module` or `None`, optional
        A differentiable model that tells us how to plot ``data``. See
        above for behavior if ``None``.
    batch_idx : int, optional
        Which index to take from the batch dimension

    Returns
    -------
    artists : `list`
        A list of the artists used to update the information on the
        plots

    """
    if isinstance(data, dict):
        for v in data.values():
            if v.ndim not in [3, 4]:
                raise ValueError(
                    "update_plot expects 3 or 4 dimensional data"
                    "; unexpected behavior will result otherwise!"
                    f" Got data of shape {v.shape}"
                )
    else:
        if data.ndim not in [3, 4]:
            raise ValueError(
                "update_plot expects 3 or 4 dimensional data"
                "; unexpected behavior will result otherwise!"
                f" Got data of shape {data.shape}"
            )
    try:
        artists = model.update_plot(axes=axes, batch_idx=batch_idx, data=data)
    except AttributeError:
        ax_artists = _get_artists_from_axes(axes, data)
        artists = []
        if not isinstance(data, dict):
            data_dict = {}
            # check for RGBA images
            if len(ax_artists) == 1 and data.shape[1] > 1:
                # can't index into dict.values(), so use this work around
                # instead, as suggested
                # https://stackoverflow.com/questions/43629270/how-to-get-single-value-from-dict-with-single-entry
                try:
                    if next(iter(ax_artists.values())).get_array().data.ndim > 1:
                        # then this is an RGBA image
                        data_dict = {"00": data}
                except Exception as e:
                    raise Exception(
                        "Thought this was an RGB(A) image based on the number"
                        " of artists and data shape, but something is off!"
                        f" Original exception: {e}"
                    )
            else:
                for i, d in enumerate(data.unbind(1)):
                    # need to keep the shape the same because of how we
                    # check for shape below (unbinding removes a dimension,
                    # so we add it back)
                    data_dict[f"{i:02d}"] = d.unsqueeze(1)
            data = data_dict
        for k, d in data.items():
            try:
                art = ax_artists[k]
            except KeyError:
                # If the we're grabbing these labels from the line labels and
                # they were originally ints, they will get converted to
                # strings. this catches that
                art = ax_artists[str(k)]
            d = to_numpy(d[batch_idx]).squeeze()
            if d.ndim == 1:
                try:
                    # then it's a line
                    x, _ = art.get_data()
                    art.set_data(x, d)
                    artists.append(art)
                except AttributeError:
                    # then it's a stemplot
                    sc = update_stem(art, d)
                    artists.extend([sc.markerline, sc.stemlines])
            elif d.ndim == 2:
                try:
                    # then it's a grayscale image
                    art.set_data(d)
                    artists.append(art)
                except AttributeError:
                    # then it's a scatterplot
                    art.set_offsets(d)
                    artists.append(art)
            else:
                # then it's an RGB(A) image. for tensors, we put that dimension
                # in channel, but for images, it should be at the end
                art.set_data(np.moveaxis(d, 0, -1))
                artists.append(art)
    # make sure to always return a list
    if not isinstance(artists, list):
        artists = [artists]
    return artists


def plot_representation(
    model=None,
    data=None,
    ax=None,
    figsize=(5, 5),
    ylim=False,
    batch_idx=0,
    title="",
    as_rgb=False,
):
    r"""Helper function for plotting model representation

    We are trying to plot ``data`` on ``ax``, using
    ``model.plot_representation`` method, if it has it, and otherwise
    default to a function that makes sense based on the shape of ``data``.

    All of these arguments are optional, but at least some of them need
    to be set:

    - If ``model`` is ``None``, we fall-back to a type of plot based on the
      shape of ``data``. If it looks image-like, we'll use ``plenoptic.imshow``
      and if it looks vector-like, we'll use ``plenoptic.clean_stem_plot``. If
      it's a dictionary, we'll assume each key, value pair gives the title and
      data to plot on a separate sub-plot.

    - If ``data`` is ``None``, we can only do something if
      ``model.plot_representation`` has some default behavior when
      ``data=None``; this is probably to plot its own ``representation``
      attribute. Thus, this will raise an Exception if both ``model`` and
      ``data`` are ``None``, because we have no idea what to plot then.

    - If ``ax`` is ``None``, we create a one-subplot figure using ``figsize``.
      If ``ax`` is not ``None``, we therefore ignore ``figsize``.

    - If ``ylim`` is ``None``, we call ``rescale_ylim``, which sets the axes'
      y-limits to be ``(-y_max, y_max)``, where ``y_max=np.abs(data).max()``.
      If it's ``False``, we do nothing.

    Parameters
    ----------
    model : `torch.nn.Module` or None, optional
        A differentiable model that tells us how to plot ``data``. See
        above for behavior if ``None``.
    data : `array_like`, `dict`, or `None`, optional
        The data to plot. See above for behavior if ``None``.
    ax : matplotlib.pyplot.axis or None, optional
        The axis to plot on. See above for behavior if ``None``.
    figsize : `tuple`, optional
        The size of the figure to create. Ignored if ``ax`` is not
        ``None``.
    ylim : `tuple`, `None`, or `False`, optional
        If not None, the y-limits to use for this plot. See above for
        behavior if ``None``. If False, we do nothing.
    batch_idx : `int`, optional
        Which index to take from the batch dimension
    title : `str`, optional
        The title to put above this axis. If you want no title, pass
        the empty string (``''``)
    as_rgb : bool, optional
        The representation can be image-like with multiple channels, and we
        have no way to determine whether it should be represented as an RGB
        image or not, so the user must set this flag to tell us. It will be
        ignored if the representation doesn't look image-like or if the
        model has its own plot_representation_error() method. Else, it will
        be passed to `po.imshow()`, see that methods docstring for details.

    Returns
    -------
    axes : list
        List of created axes.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        warnings.warn("ax is not None, so we're ignoring figsize...")
        fig = ax.figure
    try:
        # no point in passing figsize, because we've already created
        # and are passing an axis or are passing the user-specified one
        fig, axes = model.plot_representation(
            ylim=ylim, ax=ax, title=title, batch_idx=batch_idx, data=data
        )
    except AttributeError:
        if data is None:
            data = model.representation
        if not isinstance(data, dict):
            if title is None:
                title = "Representation"
            data_dict = {}
            if not as_rgb:
                # then we peel apart the channels
                for i, d in enumerate(data.unbind(1)):
                    # need to keep the shape the same because of how we
                    # check for shape below (unbinding removes a dimension,
                    # so we add it back)
                    data_dict[title + f"_{i:02d}"] = d.unsqueeze(1)
            else:
                data_dict[title] = data
            data = data_dict
        else:
            warnings.warn("data has keys, so we're ignoring title!")
        # want to make sure the axis we're taking over is basically invisible.
        ax = clean_up_axes(ax, False, ["top", "right", "bottom", "left"], ["x", "y"])
        axes = []
        if len(list(data.values())[0].shape) == 3:
            # then this is 'vector-like'
            gs = ax.get_subplotspec().subgridspec(
                min(4, len(data)), int(np.ceil(len(data) / 4))
            )
            for i, (k, v) in enumerate(data.items()):
                ax = fig.add_subplot(gs[i % 4, i // 4])
                # only plot the specified batch, but plot each channel
                # in a separate call. there should probably only be one,
                # and if there's not you probably want to do things
                # differently
                for d in v[batch_idx]:
                    ax = clean_stem_plot(to_numpy(d), ax, k, ylim)
                axes.append(ax)
        elif len(list(data.values())[0].shape) == 4:
            # then this is 'image-like'
            gs = ax.get_subplotspec().subgridspec(
                int(np.ceil(len(data) / 4)), min(4, len(data))
            )
            for i, (k, v) in enumerate(data.items()):
                ax = fig.add_subplot(gs[i // 4, i % 4])
                ax = clean_up_axes(
                    ax, False, ["top", "right", "bottom", "left"], ["x", "y"]
                )
                # only plot the specified batch
                imshow(
                    v,
                    batch_idx=batch_idx,
                    title=k,
                    ax=ax,
                    vrange="indep0",
                    as_rgb=as_rgb,
                )
                axes.append(ax)
            # because we're plotting image data, don't want to change
            # ylim at all
            ylim = False
        else:
            raise Exception(f"Don't know what to do with data of shape {data.shape}")
    if ylim is None:
        if isinstance(data, dict):
            data = torch.cat(list(data.values()), dim=2)
        rescale_ylim(axes, data)
    return axes
