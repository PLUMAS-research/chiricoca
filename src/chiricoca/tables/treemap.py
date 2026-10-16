import warnings

import matplotlib.pyplot as plt
import numpy as np
import squarify
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

from chiricoca.colors import categorical_color_legend, categorical_colors


def treemap(
    df,
    levels,
    value,
    hue=None,
    palette=None,
    aspect=None,
    padding=0.006,
    edgecolor="white",
    linewidth=0.5,
    group_edgecolor="#333333",
    group_linewidth=1.2,
    labels=False,
    label_args=None,
    group_labels=False,
    group_label_args=None,
    legend=False,
    legend_args=None,
    ax=None,
    fig_args=None,
):
    """
    Treemap anidado: un rectángulo por hoja, con área proporcional a `value`,
    dentro del rectángulo de cada grupo de los niveles superiores.

    Cada nivel se reparte con el algoritmo squarified de Bruls, Huizing y van
    Wijk (2000), que busca rectángulos lo más cuadrados posible. Los grupos y
    las hojas se ordenan de mayor a menor, así que el más grande de cada nivel
    queda arriba a la izquierda.

    Parameters
    ----------
    df : DataFrame
        Una fila por hoja. Si varias filas comparten los mismos `levels`, sus
        valores se suman.
    levels : list of str
        Columnas que definen la jerarquía, del primer nivel a las hojas.
    value : str
        Columna con el atributo cuantitativo, que debe ser positivo. Las filas
        con valor cero o negativo se descartan con un aviso.
    hue : str o None
        Columna que colorea las hojas. Tiene que ser constante dentro de cada
        hoja. Por omisión, el primer nivel.
    palette : str, list o dict
        Paleta de `categorical_colors`; un diccionario fija el color de cada
        categoría.
    aspect : float o None
        Alto dividido por ancho del rectángulo completo. Por omisión, el de los
        ejes, para que el treemap los llene.
    padding : float
        Margen entre el borde de un grupo y sus hijos, como fracción del ancho.
    edgecolor, linewidth : borde de cada hoja.
    group_edgecolor, group_linewidth : borde de cada grupo. El ancho baja a la
        mitad en cada nivel.
    labels : bool o dict
        Rotular las hojas con su nombre (`True`) o con el texto de un
        diccionario hoja -> texto. El rótulo se escribe solo si cabe: primero
        horizontal, después vertical y después con letra más chica, hasta
        `min_fontsize`.
    label_args : dict o None
        `fontsize` (6), `min_fontsize` (4.5) y `color` (por omisión, blanco
        sobre los colores oscuros y casi negro sobre los claros).
    group_labels : bool o dict
        Rotular los grupos del primer nivel, en la esquina superior izquierda,
        con su nombre o con el texto de un diccionario grupo -> texto.
    group_label_args : dict o None
        `fontsize` (7) y `color` ("white", con un contorno del color de
        `group_edgecolor`).
    legend : bool
        Agregar la leyenda de `hue` con `categorical_color_legend`, con las
        categorías de mayor a menor total.
    legend_args : dict o None
        Argumentos de `categorical_color_legend`.
    ax : matplotlib.axes.Axes o None
        Ejes donde dibujar. Si es None, se crea una figura.
    fig_args : dict o None
        Argumentos de `plt.subplots` si `ax` es None.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ax = treemap(censo, levels=["provincia", "comuna"], value="poblacion",
    ...              labels=True, legend=True)
    """
    if ax is None:
        fig, ax = plt.subplots(**(fig_args or {"figsize": (7.4, 3.8)}))
    fig = ax.get_figure()

    levels = list(levels)
    hue = hue or levels[0]
    positivos = df[value] > 0
    if not positivos.all():
        warnings.warn(f"Se descartan {(~positivos).sum()} filas con valor cero o negativo.",
                      stacklevel=2)
    columnas = list(dict.fromkeys(levels + [hue]))
    hojas = df.loc[positivos, columnas + [value]]
    if hue not in levels and (hojas.groupby(levels, observed=True)[hue].nunique() > 1).any():
        raise ValueError(f"`{hue}` tiene más de un valor dentro de una hoja")
    hojas = hojas.groupby(columnas, observed=True, sort=False)[value].sum().reset_index()
    colores, _ = categorical_colors(df[hue], palette)

    if aspect is None:
        caja = ax.get_position()
        ancho, alto = fig.get_size_inches()
        aspect = (caja.height * alto) / (caja.width * ancho)

    rectangulos = []

    def repartir(tabla, nivel, x, y, dx, dy):
        columna = levels[nivel]
        totales = (tabla.groupby(columna, observed=True, sort=False)[value].sum()
                   .sort_values(ascending=False))
        partes = squarify.squarify(squarify.normalize_sizes(totales.to_numpy(), dx, dy),
                                   x, y, dx, dy)
        for nombre, r in zip(totales.index, partes):
            grupo = tabla[tabla[columna] == nombre]
            if nivel == len(levels) - 1:
                color = colores[grupo[hue].iloc[0]]
                rectangulos.append((nombre, color, r))
                ax.add_patch(Rectangle((r["x"], r["y"]), r["dx"], r["dy"], facecolor=color,
                                       edgecolor=edgecolor, linewidth=linewidth))
                continue
            margen = min(padding, r["dx"] / 4, r["dy"] / 4)
            repartir(grupo, nivel + 1, r["x"] + margen, r["y"] + margen,
                     r["dx"] - 2 * margen, r["dy"] - 2 * margen)
            ax.add_patch(Rectangle((r["x"], r["y"]), r["dx"], r["dy"], facecolor="none",
                                   edgecolor=group_edgecolor,
                                   linewidth=group_linewidth * 0.5 ** nivel, zorder=3))
            if nivel == 0 and group_labels is not False:
                texto = group_labels.get(nombre, nombre) if isinstance(group_labels, dict) else nombre
                args = {"fontsize": 7, "color": "white", **(group_label_args or {})}
                ax.text(r["x"] + 2 * margen, r["y"] + 2 * margen, texto, ha="left", va="top",
                        zorder=4, path_effects=[withStroke(linewidth=2, foreground=group_edgecolor)],
                        **args)

    repartir(hojas, 0, 0, 0, 1, aspect)

    # El eje y va invertido para que el primer rectángulo quede arriba.
    ax.set_xlim(0, 1)
    ax.set_ylim(aspect, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if legend:
        # La leyenda sigue el orden de los rectángulos, de mayor a menor.
        orden = hojas.groupby(hue, observed=True)[value].sum().sort_values(ascending=False).index
        categorical_color_legend(ax, {k: colores[k] for k in orden}, **(legend_args or {}))

    if labels is not False:
        _rotular_hojas(ax, rectangulos, labels, label_args or {})

    return ax


def _rotular_hojas(ax, rectangulos, labels, label_args):
    """Escribe el rótulo de cada hoja solo donde cabe, medido en pantalla."""
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tamano = label_args.get("fontsize", 6)
    minimo = label_args.get("min_fontsize", 4.5)

    for nombre, color, r in rectangulos:
        texto = labels.get(nombre) if isinstance(labels, dict) else str(nombre)
        if not texto:
            continue
        (x0, y0), (x1, y1) = ax.transData.transform(
            [(r["x"], r["y"]), (r["x"] + r["dx"], r["y"] + r["dy"])]
        )
        ancho, alto = abs(x1 - x0) * 0.9, abs(y1 - y0) * 0.9
        luminancia = np.dot(to_rgb(color), [0.299, 0.587, 0.114])
        rotulo = ax.text(r["x"] + r["dx"] / 2, r["y"] + r["dy"] / 2, texto, ha="center",
                         va="center", color=label_args.get("color")
                         or ("#222222" if luminancia > 0.5 else "white"))
        for fontsize in np.arange(tamano, minimo - 0.01, -0.5):
            rotulo.set_fontsize(fontsize)
            for rotacion in (0, 90) if alto > ancho else (0,):
                rotulo.set_rotation(rotacion)
                caja = rotulo.get_window_extent(renderer)
                if caja.width <= ancho and caja.height <= alto:
                    break
            else:
                continue
            break
        else:
            rotulo.remove()
