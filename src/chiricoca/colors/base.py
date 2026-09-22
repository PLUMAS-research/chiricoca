import warnings

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colorbar
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class MidpointNormalize(colors.TwoSlopeNorm):
    """Normalización divergente centrada en `midpoint`.

    Es `matplotlib.colors.TwoSlopeNorm` con el nombre histórico de chiricoca:
    la mitad de la paleta queda en `midpoint`, con pendientes distintas a cada
    lado si `vmin` y `vmax` no son simétricos.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=0.0, clip=False):
        super().__init__(vcenter=midpoint, vmin=vmin, vmax=vmax)

    @property
    def midpoint(self):
        return self.vcenter


def colormap_from_palette(palette_name, n_colors=10):
    """`ListedColormap` de `n_colors` colores a partir de una paleta de seaborn o de un `Colormap`.

    Una paleta cualitativa (`tab10`, `Set2`, `deep`) tiene una cantidad fija de
    colores; pedir más los repite en ciclo, y un mapa continuo hecho con ella
    muestra bandas sin sentido. En ese caso se avisa.
    """
    if isinstance(palette_name, colors.Colormap):
        paleta = [palette_name(v) for v in np.linspace(0, 1, n_colors)]
    else:
        paleta = sns.color_palette(palette_name, n_colors=n_colors)
    distintos = len({tuple(np.round(c[:3], 6)) for c in paleta})
    if n_colors > 1 and distintos < len(paleta):
        warnings.warn(
            f"La paleta {palette_name!r} tiene {distintos} colores distintos y se pidieron "
            f"{n_colors}: los colores se repiten en ciclo. Para una escala continua usar una "
            "paleta secuencial o divergente.",
            stacklevel=2,
        )
    return colors.ListedColormap(paleta)


def inset_colorbar_axes(
    ax,
    location="lower center",
    width="50%",
    height="5%",
    orientation="horizontal",
    bbox_to_anchor=(0.0, 0.0, 0.95, 0.95),
    bbox_transform=None,
    borderpad=None,
    pad=0.05,
):
    """Eje para una barra de color, inserto en `ax` o pegado a su borde.

    Con cualquier `location` de matplotlib el eje queda dentro de `ax`, con el
    tamaño dado como fracción del eje o en pulgadas. Con `location="out"` se
    pega debajo (horizontal) o a la derecha (vertical), cediendo espacio del
    mapa con `make_axes_locatable`, y `width` y `height` dimensionan la barra
    dentro de esa franja. Es el eje que usan `add_ranged_color_legend`,
    `heat_map` y `flow_map`.
    """
    if bbox_transform is None:
        bbox_transform = ax.transAxes
    extra = {} if borderpad is None else {"borderpad": borderpad}
    if location != "out":
        return inset_axes(
            ax, width=width, height=height, loc=location,
            bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform, **extra,
        )
    divider = make_axes_locatable(ax)
    franja = divider.append_axes(
        "bottom" if orientation == "horizontal" else "right",
        size=height if orientation == "horizontal" else width,
        pad=pad,
    )
    franja.set_axis_off()
    return inset_axes(
        franja, width=width, height=height, loc="center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=franja.transAxes, borderpad=0,
    )


def fit_tick_labels(cax, labels, width_inches, fontsize="x-small", rotation=45):
    """Rota las etiquetas del eje x de `cax` cuando, puestas en fila, no caben en `width_inches`.

    El ancho de las etiquetas se estima con 0,6 veces el tamaño de fuente por
    carácter, que basta para decidir; la rotación deja las etiquetas alineadas
    por su extremo derecho al tick.
    """
    from matplotlib.font_manager import FontProperties

    puntos = FontProperties(size=fontsize).get_size_in_points()
    if sum(len(str(e)) for e in labels) * 0.6 * puntos / 72 > width_inches:
        cax.tick_params(axis="x", labelrotation=rotation)
        for texto in cax.get_xticklabels():
            texto.set_ha("right")
    return cax


def _inches(size, ax):
    """Ancho en pulgadas de un tamaño de eje inserto: fracción del eje padre (`"50%"`) o pulgadas."""
    ancho_padre = ax.get_position().width * ax.figure.get_size_inches()[0]
    if isinstance(size, str) and size.endswith("%"):
        return ancho_padre * float(size[:-1]) / 100
    return float(size)


def format_edge(x):
    """Etiqueta compacta para un borde de clase: sin decimales desde diez, dos cifras significativas abajo."""
    return f"{x:,.0f}" if abs(x) >= 10 else f"{x:.2g}"


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
    """Dibuja en `ax` una leyenda de clases de color.

    Sin `sizes` es una barra de color por tramos con un tick en cada borde de
    `bins` (`len(bins)` debe ser la cantidad de colores más uno). Con
    `bin_spacing="uniform"` cada clase ocupa el mismo largo, que es lo
    legible cuando las clases vienen de cuantiles; con `"proportional"` el
    largo sigue el ancho del intervalo. Con `sizes`, una cantidad por clase,
    la leyenda pasa a ser un histograma: barras del color de cada clase con
    la altura de `sizes`, que muestra cuántas unidades caen en cada una.
    `tick_labels` reemplaza los números de los bordes.
    """
    if bins is None:
        if isinstance(color_list, colors.ListedColormap):
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
        if isinstance(color_list, colors.ListedColormap):
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


_SRGB_A_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
_BLANCO_D65 = np.array([0.95047, 1.0, 1.08883])


def _srgb_to_lab(rgb):
    """sRGB en [0, 1] a CIE Lab con blanco D65, la misma convención que usaba `spectra`."""
    rgb = np.asarray(rgb, dtype=float)
    lineal = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    t = (_SRGB_A_XYZ @ lineal) / _BLANCO_D65
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _lab_to_srgb(lab):
    """Inversa de `_srgb_to_lab`; el resultado puede salir de [0, 1] y hay que recortarlo."""
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    t = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))
    lineal = np.clip(np.linalg.inv(_SRGB_A_XYZ) @ (t * _BLANCO_D65), 0, None)
    return np.where(lineal <= 0.0031308, 12.92 * lineal, 1.055 * lineal ** (1 / 2.4) - 0.055)


def _ajustar_lch(rgb, luminancia=0.0, croma=0.0):
    """Suma `luminancia` a L y `croma` a C en CIE LCH, y vuelve a sRGB."""
    L, a, b = _srgb_to_lab(rgb)
    C, H = np.hypot(a, b) + croma, np.arctan2(b, a)
    return _lab_to_srgb((L + luminancia, C * np.cos(H), C * np.sin(H)))


def bivariate_matrix_from_palette(
    palette_name="PiYG", n_colors=3, darken_initial_value=-15
):
    """Matriz `n_colors` por `n_colors` por 3 de colores para una coropleta bivariada.

    Con un nombre de paleta divergente de seaborn, cada eje toma una mitad de
    la paleta y las celdas interiores mezclan ambas, aclaradas y saturadas en
    CIE Lab y LCH. Los colores quedan recortados a `[0, 1]`. Una matriz ya
    construida (lista anidada de colores hex o RGB, o arreglo con forma
    `(n_colors, n_colors, 3)`, indexada `[fila_y][columna_x]` con la fila 0
    abajo) se valida y devuelve tal cual: así se usan `BIVARIATE_BLUE_ORANGE`
    y `BIVARIATE_PURPLE_TEAL`, las dos matrices accesibles del módulo.
    """
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
    cmap_x = [np.asarray(c[:3]) for c in full_palette[n_colors - 1 :]]
    cmap_y = [np.asarray(c[:3]) for c in list(reversed(full_palette))[n_colors - 1 :]]

    cmap_xy = []
    for j in range(n_colors):
        for i in range(n_colors):
            x, y = cmap_x[i], cmap_y[j]
            if i == j and i == 0:
                cmap_xy.append(_ajustar_lch(x, luminancia=-darken_initial_value))
            elif i == 0:
                cmap_xy.append(y)
            elif j == 0:
                cmap_xy.append(x)
            else:
                # mezcla en sRGB, aclarada 1,5 en L y, fuera de la diagonal,
                # saturada 3 en C
                mezcla = (x + y) / 2
                cmap_xy.append(_ajustar_lch(mezcla, luminancia=1.5, croma=0.0 if i == j else 3.0))

    return np.clip(np.array(cmap_xy).reshape(n_colors, n_colors, 3), 0.0, 1.0)


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
        # un color casi blanco desaparece sobre el fondo de la leyenda; lleva
        # un borde gris fino
        r, g, b = colors.to_rgb(color)
        claro = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.85
        legend_elements.append(
            Patch(facecolor=color, edgecolor="0.6" if claro else "none", linewidth=0.5, label=label)
        )

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
    """Leyenda de clases de color como eje inserto en `ax`.

    `bins` son los bordes de las clases (uno más que los colores de
    `built_palette`). Los argumentos que sobran van a `color_legend`: por
    ejemplo `bin_spacing="uniform"` para que cada clase ocupe lo mismo, o
    `sizes` para dibujar la leyenda como histograma de clases. `location="out"`
    la pega fuera del mapa. Devuelve el eje de la leyenda (`cbar_ax` lo
    reemplaza por uno propio).
    """
    cbar_ax = kwargs.pop("cbar_ax", None)
    if cbar_ax is None:
        cbar_ax = inset_colorbar_axes(
            ax, location=location, width=width, height=height, orientation=orientation,
            bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform,
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
    if orientation == "horizontal":
        etiquetas = tick_labels if tick_labels else [f"{b:g}" for b in bins]
        fit_tick_labels(cbar_ax, etiquetas, _inches(width, ax), label_size)

    return cbar_ax
