import numpy as np
import matplotlib.pyplot as plt
import contextily as cx


def _panel_aspect_ratio(geodf, bbox, aspect):
    if aspect == "auto":
        if geodf.crs and geodf.crs.is_geographic:
            y_coord = (bbox[1] + bbox[3]) / 2
            geo_aspect = 1 / np.cos(y_coord * np.pi / 180)
        else:
            geo_aspect = 1.0
    else:
        geo_aspect = aspect
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    return (bbox_w / bbox_h) / geo_aspect, geo_aspect


def set_axis_aspect(ax, geodf):
    # code from geopandas
    if geodf.crs and geodf.crs.is_geographic:
        bounds = geodf.total_bounds
        y_coord = np.mean([bounds[1], bounds[3]])
        ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))
        # formula ported from R package sp
        # https://github.com/edzer/sp/blob/master/R/mapasp.R
    else:
        ax.set_aspect("equal")

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

        set_axis_aspect(ax, geodf)

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


def small_multiples_from_geodataframes(
    geodfs,
    height=5,
    col_wrap=5,
    bboxes=None,
    remove_axes=True,
    set_limits=True,
    flatten_axes=True,
    aspect="auto",
    basemap=None,
    basemap_interpolation="hanning",
):
    """Small multiples donde cada panel usa el bbox y el aspect ratio de su propio geodataframe.

    A diferencia de `small_multiples_from_geodataframe`, los paneles no
    comparten xlim/ylim y el ancho de cada columna es proporcional al
    aspect ratio geografico del panel. Util cuando se comparan geografias
    de forma o extension distintas.

    Parametros
    ----------
    geodfs : list of GeoDataFrame
        Un geodataframe por panel.
    bboxes : list of array-like, opcional
        Bounding boxes por panel ``(minx, miny, maxx, maxy)``. Por defecto
        se usa el ``total_bounds`` de cada geodataframe.

    Notas
    -----
    En layouts de varias filas, el ancho de cada columna es el maximo
    aspect ratio entre los paneles de esa columna. Los paneles mas angostos
    que la columna mostraran espacio en blanco a los lados.
    """
    n = len(geodfs)
    if n == 0:
        raise ValueError("geodfs must contain at least one GeoDataFrame")

    if bboxes is None:
        bboxes = [gdf.total_bounds for gdf in geodfs]
    elif len(bboxes) != n:
        raise ValueError("bboxes must have the same length as geodfs")

    panel_aspects = []
    geo_aspects = []
    for gdf, bbox in zip(geodfs, bboxes):
        pa, ga = _panel_aspect_ratio(gdf, bbox, aspect)
        panel_aspects.append(pa)
        geo_aspects.append(ga)

    n_columns = min(col_wrap, n)
    n_rows = (n + n_columns - 1) // n_columns

    col_widths = [0.0] * n_columns
    for i, pa in enumerate(panel_aspects):
        col = i % n_columns
        col_widths[col] = max(col_widths[col], pa)

    fig_width = sum(col_widths) * height
    fig_height = n_rows * height

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": col_widths},
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    flattened = axes.flatten()

    for i in range(n):
        ax = flattened[i]
        gdf = geodfs[i]
        bbox = bboxes[i]

        if set_limits:
            ax.set_xlim([bbox[0], bbox[2]])
            ax.set_ylim([bbox[1], bbox[3]])

        ax.set_aspect(geo_aspects[i])

        if remove_axes:
            ax.set_axis_off()

        if basemap is not None:
            cx.add_basemap(
                ax,
                crs=gdf.crs.to_string(),
                source=basemap,
                interpolation=basemap_interpolation,
                zorder=0,
            )

    for j in range(n, len(flattened)):
        flattened[j].set_axis_off()

    if flatten_axes:
        return fig, flattened

    return fig, axes
