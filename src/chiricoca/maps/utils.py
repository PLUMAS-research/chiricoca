import matplotlib.patheffects as path_effects
from matplotlib_scalebar.scalebar import (
    ScaleBar,
)
import contextily as cx


def geographical_labels(
    ax,
    geodf,
    column=None,
    font_size=11,
    font_weight="bold",
    color="white",
    outline=True,
    outline_color="black",
    outline_width=2,
    format_func=None,
    bounds=None,
    displacements=None,
    va="center",
    ha="center",
    anchor="centroid",
):
    """Rotula cada geometría de `geodf` con su índice o con `column`.

    `anchor` decide dónde va el rótulo: `"centroid"` usa el centroide
    calculado en un CRS proyectado (`projected_centroids`), que en un polígono
    cóncavo puede caer fuera de él; `"representative"` usa
    `representative_point`, que siempre cae dentro. `displacements` es un
    desplazamiento común `(dx, dy)` o un diccionario etiqueta a desplazamiento.
    Devuelve la lista de textos, que sirve para pasarla a `adjust_text`.
    """
    from chiricoca.geo.utils import projected_centroids

    puntos = (
        geodf.geometry.representative_point() if anchor == "representative"
        else projected_centroids(geodf.geometry)
    )
    etiquetas = geodf.index if column is None else geodf[column]

    labels = []
    for etiqueta, x, y in zip(etiquetas, puntos.x.to_numpy(), puntos.y.to_numpy()):
        if bounds is not None and not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
            continue
        if isinstance(displacements, dict):
            dx, dy = displacements.get(etiqueta, (0, 0))
        else:
            dx, dy = displacements if displacements is not None else (0, 0)
        texto = format_func(etiqueta) if format_func is not None else etiqueta

        t = ax.text(x + dx, y + dy, texto, va=va, ha=ha, fontsize=font_size, fontweight=font_weight, color=color)
        if outline:
            t.set_path_effects(
                [path_effects.Stroke(linewidth=outline_width, foreground=outline_color), path_effects.Normal()]
            )
        labels.append(t)

    return labels


def north_arrow(
    ax,
    x=0.98,
    y=0.06,
    arrow_length=0.04,
    text="N",
    font_name=None,
    font_size=None,
    color="#000000",
    arrow_color="#000000",
    arrow_width=3,
    arrow_headwidth=7,
):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(
            facecolor=arrow_color, width=arrow_width, headwidth=arrow_headwidth
        ),
        ha="center",
        va="center",
        fontsize=font_size,
        fontname=font_name,
        color=color,
        xycoords=ax.transAxes,
    )


def geographical_scale(ax, *args, **kwargs):
    ax.add_artist(ScaleBar(*args, **kwargs))


def context_boundaries(
    ax, geodf, color="black", linewidth=0.5, alpha=0.3, zorder=None
):
    """Dibuja los bordes de un GeoDataFrame como capa de contexto sobre un
    mapa existente (por ejemplo, límites administrativos sobre una coropleta),
    preservando el encuadre actual del eje.

    Parameters
    ----------
    ax : matplotlib Axes
        Eje con el mapa ya dibujado.
    geodf : GeoDataFrame
        Polígonos cuyos bordes se dibujarán (se usa `.boundary`).
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    geodf.boundary.plot(
        ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def add_basemap(
    ax,
    file_path,
    geocontext,
    interpolation="hanning",
    zorder=0,
    reset_extent=False,
    **kwargs
):
    bounds = geocontext.total_bounds
    cx.add_basemap(
        ax,
        crs=geocontext.crs.to_string(),
        source=file_path,
        interpolation=interpolation,
        zorder=zorder,
        reset_extent=reset_extent,
        **kwargs
    )

    if not reset_extent:
        ax.set_xlim((bounds[0], bounds[2]))
        ax.set_ylim((bounds[1], bounds[3]))
