import geopandas as gpd
from chiricoca.geo.figures import figure_from_geodataframe


def bubble_map(
    geodf: gpd.GeoDataFrame,
    size,
    scale=1,
    color=None,
    add_legend=True,
    edgecolor="white",
    alpha=1.0,
    label=None,
    ax=None,
    fig_args=None,
    **kwargs
):
    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = figure_from_geodataframe(geodf, **fig_args)

    marker = "o"

    if size is not None:
        if type(size) == str:
            marker_size = geodf[size]
        else:
            marker_size = float(size)
    else:
        marker_size = 1

    geodf.plot(
        ax=ax,
        marker=marker,
        markersize=marker_size * scale,
        edgecolor=edgecolor,
        alpha=alpha,
        facecolor=color,
        legend=add_legend,
        label=label,
        **kwargs
    )

    return ax


def dot_map(geodf: gpd.GeoDataFrame, size=10, add_legend=True, label=None, **kwargs):
    return bubble_map(
        geodf,
        size=float(size),
        add_legend=add_legend,
        edgecolor="none",
        label=label,
        **kwargs
    )
