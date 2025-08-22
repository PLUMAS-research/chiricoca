import numpy as np
import matplotlib.pyplot as plt
import contextily as cx


def figure_from_geodataframe(
    geodf,
    height=5,
    bbox=None,
    remove_axes=True,
    set_limits=True,
    basemap=None,
    basemap_interpolation="hanning",
    **kwargs
):
    if bbox is None:
        bbox = geodf.total_bounds

    aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    fig, ax = plt.subplots(figsize=(height * aspect, height), **kwargs)

    if set_limits:
        ax.set_xlim([bbox[0], bbox[2]])
        ax.set_ylim([bbox[1], bbox[3]])

        # code from geopandas
        if geodf.crs and geodf.crs.is_geographic:
            bounds = geodf.total_bounds
            y_coord = np.mean([bounds[1], bounds[3]])
            ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
            # formula ported from R package sp
            # https://github.com/edzer/sp/blob/master/R/mapasp.R
        else:
            ax.set_aspect("equal")

    if remove_axes:
        ax.set_axis_off()

    if basemap is not None:
        cx.add_basemap(
            ax,
            crs=geodf.crs.to_string(),
            source=basemap,
            interpolation=basemap_interpolation,
            zorder=0,
        )

    return fig, ax


def small_multiples_from_geodataframe(
    geodf,
    n_variables,
    height=5,
    col_wrap=5,
    bbox=None,
    sharex=True,
    sharey=True,
    remove_axes=True,
    set_limits=True,
    flatten_axes=True,
    aspect="auto",
    basemap=None,
    basemap_interpolation="hanning",
):
    if n_variables <= 1:
        return figure_from_geodataframe(
            geodf,
            height=height,
            bbox=bbox,
            remove_axes=remove_axes,
            set_limits=set_limits,
            basemap=basemap,
            basemap_interpolation=basemap_interpolation,
        )

    if bbox is None:
        bbox = geodf.total_bounds

    # code from geopandas
    if aspect == "auto":
        if geodf.crs and geodf.crs.is_geographic:
            y_coord = np.mean([bbox[1], bbox[3]])
            aspect_ratio = 1 / np.cos(y_coord * np.pi / 180)
            # formula ported from R package sp
            # https://github.com/edzer/sp/blob/master/R/mapasp.R
        else:
            aspect_ratio = 1
    else:
        aspect_ratio = aspect

    n_columns = min(col_wrap, n_variables)
    n_rows = n_variables // n_columns
    if n_rows * n_columns < n_variables:
        n_rows += 1

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(n_columns * height / aspect_ratio, n_rows * height),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    flattened = axes.flatten()

    if set_limits:
        for ax in flattened:
            ax.set_xlim([bbox[0], bbox[2]])
            ax.set_ylim([bbox[1], bbox[3]])

    for ax in flattened:
        ax.set_aspect(aspect_ratio)

    if remove_axes:
        for ax in flattened:
            ax.set_axis_off()
    else:
        # deactivate only unneeded axes
        for i in range(n_variables, len(axes)):
            flattened[i].set_axis_off()

    if basemap is not None:
        for ax in flattened:
            cx.add_basemap(
                ax,
                crs=geodf.crs.to_string(),
                source=basemap,
                interpolation=basemap_interpolation,
                zorder=0,
            )

    if flatten_axes:
        return fig, flattened

    return fig, axes
