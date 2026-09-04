import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
import spectra
from matplotlib import colorbar
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from http://chris35wills.github.io/matplotlib_diverging_colorbar/


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def colormap_from_palette(palette_name, n_colors=10):
    return colors.ListedColormap(sns.color_palette(palette_name, n_colors=n_colors))


def color_legend(
    ax,
    color_list,
    bins=None,
    sizes=None,
    orientation="horizontal",
    remove_axes=False,
    bin_spacing="proportional",
    tick_labels=None,
    **kwargs,
):
    if bins is None:
        if type(color_list) == colors.ListedColormap:
            N = color_list.N
        else:
            N = len(color_list)
        bins = np.array(range(N))

    if sizes is not None:
        bar_width = bins[1:] - bins[0:-1]
        if orientation == "horizontal":
            ax.bar(
                bins[:-1],
                sizes,
                width=bar_width,
                align="edge",
                color=color_list,
                edgecolor=color_list,
            )
            ax.set_xticks(bins, labels=tick_labels, **kwargs)
        else:
            ax.barh(
                bins[:-1],
                sizes,
                height=bar_width,
                align="edge",
                color=color_list,
                edgecolor=color_list,
            )
            ax.set_yticks(bins, labels=tick_labels, **kwargs)
    else:
        cbar_norm = colors.BoundaryNorm(bins, len(bins) - 1)
        if type(color_list) == colors.ListedColormap:
            cmap = color_list
        else:
            cmap = colors.ListedColormap(color_list)
        cb = colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=cbar_norm,
            ticks=bins,
            spacing=bin_spacing,
            orientation=orientation,
            **kwargs,
        )
        if tick_labels:
            cb.set_ticklabels(tick_labels)

    sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)

    if remove_axes:
        ax.set_axis_off()

    return ax


# Paletas bivariadas 3x3 accesibles, construidas en CIELCh y validadas con
# simulacion de daltonismo (colorspacious, deuteranopia y protanopia con
# severity 100, distancias CIEDE2000; julio 2026). Indexadas [fila_y][col_x]
# con la fila 0 abajo; usarlas como argumento `palette` de
# bivariate_choropleth_map. La luminancia decrece en ambos ejes, asi que cada
# eje conserva su orden aunque se pierda el tono. Eso no las hace legibles en
# escala de grises: ahi las nueve clases dejan de separarse, porque la
# antidiagonal cruza luminancias casi iguales (L* 68, 66 y 58 en
# BIVARIATE_BLUE_ORANGE). Es inherente a una paleta bivariada.
#
# BIVARIATE_BLUE_ORANGE: eje x calido (naranja), eje y frio (azul). Distancia
# minima global entre los 9 colores: 11.7 (vision normal), 10.6 (deutan),
# 12.2 (protan), 9.2 (tritan). Los tonos se alinean con el eje amarillo-azul,
# que ambas deficiencias preservan.
BIVARIATE_BLUE_ORANGE = [
    ["#eeeeee", "#f0c3a0", "#da9768"],
    ["#b6bfec", "#bf96a0", "#af6d6c"],
    ["#6d8bc9", "#816481", "#6e384b"],
]

# BIVARIATE_PURPLE_TEAL: eje x verde-azulado, eje y purpura. Distancia minima
# adyacente: 11.6 / 9.1 / 9.3 (normal / deutan / protan), sin colisiones entre
# pares no adyacentes. Alternativa si se prefieren dos extremos frios.
BIVARIATE_PURPLE_TEAL = [
    ["#eeeeee", "#a3d7c1", "#5db69a"],
    ["#c9bae5", "#82a5b9", "#3c8995"],
    ["#8e83c5", "#45709a", "#005173"],
]


def bivariate_matrix_from_palette(
    palette_name="PiYG", n_colors=3, darken_initial_value=-15
):
    # una matriz de colores ya construida (lista anidada de colores hex o
    # RGB, o array con forma (n_colors, n_colors, 3), indexada como
    # [fila_y, columna_x]) se usa tal cual; permite paletas bivariadas
    # personalizadas como BIVARIATE_BLUE_ORANGE
    if not isinstance(palette_name, str):
        arr = np.asarray(palette_name)
        if arr.dtype.kind in ("U", "S", "O"):
            matrix = np.asarray(
                [[colors.to_rgb(c) for c in row] for row in palette_name]
            )
        else:
            matrix = arr.astype(float)
        if matrix.shape != (n_colors, n_colors, 3):
            raise ValueError(
                "la matriz de colores debe tener forma "
                f"({n_colors}, {n_colors}, 3), se recibió {matrix.shape}"
            )
        return matrix

    full_palette = sns.color_palette(palette_name, n_colors=(n_colors - 1) * 2 + 1)

    cmap_x = full_palette[n_colors - 1 :]
    cmap_y = list(reversed(full_palette))[n_colors - 1 :]

    cmap_xy = []

    for j in range(n_colors):
        for i in range(n_colors):
            x = spectra.rgb(*cmap_x[i][0:3])
            y = spectra.rgb(*cmap_y[j][0:3])

            if i == j and i == 0:
                cmap_xy.append(x.darken(darken_initial_value).rgb)

            elif i == 0:
                cmap_xy.append(y.rgb)
            elif j == 0:
                cmap_xy.append(x.rgb)
            else:
                blended = x.blend(y, ratio=0.5).brighten(1.5)

                if i == j:
                    blended = blended
                else:
                    blended = blended.saturate(3)

                cmap_xy.append(blended.rgb)

    cmap_xy = np.array(cmap_xy).reshape(n_colors, n_colors, 3)
    return cmap_xy


def categorical_color_legend(
    ax, color_list, labels=None, loc="best", n_columns=None, **kwargs
):
    """Leyenda categórica con parches de color.

    Parameters
    ----------
    ax : matplotlib Axes
    color_list : dict o iterable
        Si es dict, las claves son etiquetas y los valores colores.
        Si es iterable, se usa junto con ``labels``.
    labels : iterable, opcional
        Etiquetas correspondientes a cada color. Obligatorio si
        ``color_list`` no es dict.
    loc : str
        Posición de la leyenda.
    n_columns : int, opcional
        Cantidad de columnas en la leyenda.
    """
    if isinstance(color_list, dict):
        labels = list(color_list.keys())
        color_list = list(color_list.values())

    legend_elements = []

    for label, color in zip(labels, color_list):
        legend_elements.append(Patch(facecolor=color, edgecolor="none", label=label))

    if n_columns is not None:
        if type(n_columns) != int:
            n_columns = len(color_list)
    else:
        n_columns = 1

    previa = ax.legend_
    artist = ax.legend(
        handles=legend_elements, loc=loc, frameon=False, ncol=n_columns, **kwargs
    )
    if previa is not None and previa not in ax.get_children():
        # ax.legend reemplaza la leyenda anterior, así que se conserva como
        # artista. La nueva queda como leyenda del eje, que es la única que
        # savefig(bbox_inches="tight") considera al recortar la figura: una
        # leyenda agregada con add_artist fuera del eje queda cortada.
        ax.add_artist(previa)
    return artist


def categorical_colors(values, palette=None):
    """Asigna un color a cada categoría de ``values``.

    Parameters
    ----------
    values : Series o iterable
        Valores categóricos. Si es una Series categórica se respeta el orden
        de sus categorías; si no, las categorías se ordenan.
    palette : str, list o dict, optional
        Nombre de una paleta de seaborn, lista de colores, o diccionario
        categoría -> color, que es la forma de mantener el mismo color para
        una categoría en varias figuras. Si es None se usa ``tab10`` hasta
        diez categorías y ``husl`` con más.

    Returns
    -------
    (mapa, colores): el diccionario categoría -> color y la lista con el
    color de cada elemento de ``values``, en su orden.
    """
    serie = pd.Series(values)
    if not isinstance(serie.dtype, pd.CategoricalDtype):
        serie = serie.astype("category")
    niveles = list(serie.cat.categories)

    if isinstance(palette, dict):
        faltan = [nivel for nivel in niveles if nivel not in palette]
        if faltan:
            raise ValueError(f"La paleta no tiene color para {faltan}")
        mapa = {nivel: palette[nivel] for nivel in niveles}
    else:
        if palette is None:
            palette = "tab10" if len(niveles) <= 10 else "husl"
        mapa = dict(zip(niveles, sns.color_palette(palette, n_colors=len(niveles))))

    return mapa, [mapa[valor] for valor in serie]


def add_ranged_color_legend(
    ax,
    bins,
    built_palette,
    location="lower center",
    orientation="horizontal",
    label=None,
    label_size="x-small",
    title_size="small",
    title_align="left",
    width="50%",
    height="5%",
    bbox_to_anchor=(0.0, 0.0, 0.95, 0.95),
    bbox_transform=None,
    tick_labels=None,
    **kwargs,
):
    if bbox_transform is None:
        bbox_transform = ax.transAxes

    cbar_ax = kwargs.pop("cbar_ax", None)

    if cbar_ax is None:
        if location != "out":
            cbar_ax = inset_axes(
                ax,
                width=width,
                height=height,
                loc=location,
                bbox_to_anchor=bbox_to_anchor,
                bbox_transform=bbox_transform,
            )
        else:
            divider = make_axes_locatable(ax)
            cbar_main = divider.append_axes(
                "bottom" if orientation == "horizontal" else "right",
                size=height if orientation == "horizontal" else width,
            )
            cbar_main.set_axis_off()
            cbar_ax = inset_axes(
                cbar_main,
                width=width,
                height=height,
                loc="center",
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                bbox_transform=cbar_main.transAxes,
                borderpad=0,
            )

    color_legend(
        cbar_ax,
        built_palette,
        bins,
        orientation=orientation,
        remove_axes=False,
        tick_labels=tick_labels if tick_labels is not None else [],
        **kwargs,
    )

    sns.despine(ax=cbar_ax, left=True, top=True, bottom=True, right=True)
    if label:
        cbar_ax.set_title(label, loc=title_align, fontsize=title_size)

    cbar_ax.tick_params(labelsize=label_size)

    return cbar_ax
