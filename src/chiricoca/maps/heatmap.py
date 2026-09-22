import numpy as np

from chiricoca.colors import color_legend, colormap_from_palette, inset_colorbar_axes
from chiricoca.geo.figures import figure_from_geodataframe
from chiricoca.geo.kde import kde_from_points


def heat_map(
    geodf,
    weight=None,
    low_threshold=0,
    max_threshold=None,
    n_levels=5,
    alpha=1.0,
    palette="magma",
    kernel="cosine",
    norm=2,
    bandwidth=1e-2,
    grid_points=2**6,
    return_heat=False,
    cbar_label=None,
    cbar_width=2.4,
    cbar_height=0.15,
    cbar_location="upper left",
    cbar_orientation="horizontal",
    cbar_pad=0.05,
    cbar_bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
    cbar_bbox_transform=None,
    legend_type="none",
    ax=None,
    fig_args=None,
    **kwargs,
):
    """Mapa de calor por densidad de kernel sobre puntos, como bandas rellenas.

    `bandwidth` va en las unidades del CRS de `geodf`: en grados, 0,01 son
    cerca de un kilómetro; en un CRS métrico hay que darlo en metros. Con
    `n_levels` bandas la paleta trae un color por banda.

    Returns
    -------
    (contornos, eje_de_la_barra, ax), más la grilla de densidad `(x, y, z)`
    si `return_heat`.
    """
    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = figure_from_geodataframe(geodf, **fig_args)

    heat = kde_from_points(
        geodf,
        kernel=kernel,
        norm=norm,
        bandwidth=bandwidth,
        grid_points=grid_points,
        weight_column=weight,
    )

    norm_heat = heat[2]  # / heat[2].max()

    if type(palette) == str:
        cmap = colormap_from_palette(palette, n_colors=n_levels)
    else:
        cmap = palette

    if max_threshold is None:
        max_threshold = norm_heat.max()

    # n_levels bandas rellenas necesitan n_levels + 1 cortes, uno por color
    levels = np.linspace(low_threshold, max_threshold, n_levels + 1)

    if legend_type == "colorbar":
        cbar_ax = inset_colorbar_axes(
            ax, location=cbar_location, width=cbar_width, height=cbar_height,
            orientation=cbar_orientation, bbox_to_anchor=cbar_bbox_to_anchor,
            bbox_transform=cbar_bbox_transform, borderpad=0, pad=cbar_pad,
        )
        color_legend(cbar_ax, cmap, levels, orientation=cbar_orientation)
    else:
        cbar_ax = None

    if not return_heat:
        return (
            ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap),
            cbar_ax,
            ax,
        )
    else:
        return (
            ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap),
            cbar_ax,
            ax,
            heat,
        )
