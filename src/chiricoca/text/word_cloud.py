from random import Random
import matplotlib.pyplot as plt

import matplotlib.font_manager
import numpy as np
from cytoolz import keyfilter, valfilter
from wordcloud import WordCloud
from chiricoca.config import get_default_font_path
from chiricoca.colors.utils import truncated_mpl_colormap

class colormap_size_func(object):
    """Color func created from matplotlib colormap.
    Parameters
    ----------
    colormap : string or matplotlib colormap
        Colormap to sample from
    Example
    -------
    >>> WordCloud(color_func=colormap_color_func("magma"))
    """

    def __init__(self, colormap, max_font_size):
        import matplotlib.pyplot as plt

        self.colormap = truncated_mpl_colormap(colormap)
        self.max_font_size = max_font_size

    def __call__(
        self, word, font_size, position, orientation, random_state=None, **kwargs
    ):
        if random_state is None:
            random_state = Random()
        r, g, b, _ = 255 * np.array(self.colormap(font_size / self.max_font_size))
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)


def wordcloud(
    vocabulary,
    relative_scaling=0.5,
    size_scaling=3,
    img_scale=1,
    max_words=250,
    prefer_horizontal=1.0,
    min_font_size=8,
    max_font_size=None,
    cmap="cividis",
    background_color="white",
    mode="RGB",
    fontname=None,
    ax=None,
    fig_args=None,
    image_args={},
    **kwargs
):
    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = plt.subplots(**fig_args)

    fig_width = int(ax.get_window_extent().width * size_scaling)
    fig_height = int(ax.get_window_extent().height * size_scaling)

    if max_font_size is None:
        max_font_size = int(ax.get_window_extent().height * size_scaling * 0.66)

    if fontname is None:
        font_path = fontname = get_default_font_path()
    else:
        font_path = matplotlib.font_manager.findfont(fontname)

    wc = WordCloud(
        font_path=font_path,
        prefer_horizontal=prefer_horizontal,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        background_color=background_color,
        color_func=colormap_size_func(cmap, max_font_size),
        width=fig_width,
        height=fig_height,
        scale=img_scale,
        max_words=max_words,
        relative_scaling=relative_scaling,
        random_state=42,
        mode=mode,
    )

    wc.generate_from_frequencies(
        valfilter(lambda v: v > 0, vocabulary),
        max_font_size=max_font_size,
    )
    # wc_array = wc.to_image()

    ax.imshow(
        wc,
        interpolation="hanning",
        extent=(0, wc.width, wc.height, 0),
        aspect="auto",
        **image_args
    )

    return ax
