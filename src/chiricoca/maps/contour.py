import geopandas as gpd
import numpy as np
from shapely.geometry import box

from chiricoca.colors import colormap_from_palette
from chiricoca.geo.figures import figure_from_geodataframe


def contour_map(
    field,
    bounds,
    crs=None,
    levels=8,
    filled=False,
    palette="viridis",
    color=None,
    labels=False,
    label_format="{:.2f}",
    label_size=6,
    linewidths=0.8,
    ax=None,
    fig_args=None,
    **kwargs,
):
    """Dibuja las isolíneas de un campo escalar sobre su extensión geográfica.

    Una isolínea une los puntos donde el campo vale lo mismo, así que el mapa
    topográfico y sus curvas de nivel son el caso conocido. Sirve para cualquier
    campo continuo: vegetación, temperatura, luminosidad, densidad.

    Parameters
    ----------
    field : ndarray
        La grilla del campo, con la fila 0 arriba, que es como la entregan
        rasterio y las imágenes.
    bounds : tuple of float
        Extensión geográfica de la grilla, como (oeste, sur, este, norte), en
        las mismas unidades que el resto del mapa.
    crs : optional
        CRS de `bounds`. Solo se usa cuando la función crea la figura, para
        darle la proporción geográfica correcta con `figure_from_geodataframe`.
    levels : int or sequence of float
        Cuántas curvas dibujar, o los valores exactos de cada una. Elegir los
        valores a mano es lo que separa un mapa legible de uno que solo repite
        la escala de color.
    filled : bool
        Con True rellena entre curvas en vez de dibujar solo las líneas.
    palette : str
        Paleta con que colorear los niveles.
    color : str, optional
        Un color único para todas las curvas. Reemplaza a `palette`.
    labels : bool
        Rotula cada curva con su valor.

    Returns
    -------
    contours : matplotlib.contour.QuadContourSet

    Notes
    -----
    Las isolíneas necesitan un campo suave. Sobre una grilla ruidosa, como un
    índice de vegetación a treinta metros, cada píxel produce su propia curva y
    el mapa queda ilegible. Conviene suavizar antes, por ejemplo con
    `scipy.ndimage.gaussian_filter`, y cuánto suavizar decide qué dicen las
    curvas: es el mismo parámetro de decisión que el ancho de banda de un mapa
    de calor.
    """
    if ax is None:
        marco = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs)
        fig, ax = figure_from_geodataframe(marco, **(fig_args or {}))

    field = np.asarray(field, dtype=float)
    oeste, sur, este, norte = bounds
    # el eje y de la grilla va de arriba hacia abajo y el del mapa al revés
    extent = (oeste, este, sur, norte)

    dibujar = ax.contourf if filled else ax.contour
    opciones = dict(kwargs)
    opciones["extent"] = extent
    opciones["origin"] = "upper"
    if color is not None:
        opciones["colors"] = color
    else:
        n = levels if isinstance(levels, int) else len(levels)
        opciones["cmap"] = colormap_from_palette(palette, n_colors=max(n, 2))
    if not filled:
        opciones["linewidths"] = linewidths

    contornos = dibujar(field, levels=levels, **opciones)

    if labels and not filled:
        ax.clabel(contornos, inline=True, fontsize=label_size,
                  fmt=lambda v: label_format.format(v))

    return contornos
