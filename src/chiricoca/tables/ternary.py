import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
from adjustText import adjust_text
from ternary.helpers import project_point

from ..colors.base import categorical_color_legend, categorical_colors


def ternary_scatter(
    df,
    columns=None,
    hue=None,
    size=None,
    labels=None,
    ax=None,
    fig_args=None,
    scale=100,
    palette=None,
    color="#4c4c4c",
    size_range=(12, 240),
    alpha=0.85,
    grid_multiple=20,
    tick_multiple=20,
    tick_format="%d",
    corner_labels=True,
    corner_label_args=None,
    scatter_args=None,
    label_args=None,
    avoid_collisions=True,
    legend=True,
    legend_args=None,
    size_legend=True,
    size_legend_values=None,
    size_legend_format="{:,.0f}",
    size_legend_args=None,
):
    """Scatterplot ternario: cada fila es un punto en coordenadas baricéntricas.

    Las tres columnas se normalizan fila a fila, así que el DataFrame puede
    traer conteos o proporciones. El orden de las columnas fija los vértices:
    la primera va a la derecha, la segunda arriba y la tercera a la izquierda.

    Parameters
    ----------
    df : DataFrame
        Una fila por observación. El índice se usa como etiqueta.
    columns : list of str, optional
        Las tres columnas con las partes. Si es None, las tres primeras
        columnas numéricas que no sean ``hue`` ni ``size``.
    hue : str, optional
        Columna categórica que define el color de cada punto.
    size : str, optional
        Columna numérica que define el área de cada punto, llevada al rango
        ``size_range``.
    labels : bool or iterable, optional
        ``True`` rotula todas las filas; un iterable rotula solo los valores
        del índice que contiene.
    scale : number
        Total al que suma cada fila después de normalizar. Con 100 los ticks
        se leen como porcentajes.
    palette : str, list or dict, optional
        Paleta de seaborn para ``hue``, o diccionario categoría -> color.
    color : str
        Color de los puntos cuando no hay ``hue``.
    grid_multiple, tick_multiple : number
        Separación de las líneas de la grilla y de los ticks, en unidades de
        ``scale``.
    corner_labels : bool
        Si escribir el nombre de cada columna en su vértice.
    avoid_collisions : bool
        Si mover las etiquetas para que no se tapen entre sí (adjustText).
    legend, size_legend : bool
        Si dibujar la leyenda de color y la de tamaño.
    size_legend_values : list, optional
        Valores de ``size`` que muestra la leyenda de tamaño. Si es None, se
        usan el mínimo, la mediana y el máximo.

    Returns
    -------
    TernaryAxesSubplot. El eje de matplotlib está en ``tax.get_axes()``.
    """
    scatter_args = dict(scatter_args or {})
    label_args = {"fontsize": "x-small", **(label_args or {})}
    corner_label_args = {"fontsize": "small", **(corner_label_args or {})}
    legend_args = dict(legend_args or {})
    size_legend_args = dict(size_legend_args or {})

    if columns is None:
        columns = [
            c for c in df.columns
            if c not in (hue, size) and pd.api.types.is_numeric_dtype(df[c])
        ][:3]
    columns = list(columns)
    if len(columns) != 3:
        raise ValueError(f"Se necesitan exactamente tres columnas, se recibieron {columns}")

    partes = df[columns].astype(float)
    partes = partes.div(partes.sum(axis=1), axis=0) * scale

    if ax is None:
        fig_args = {"figsize": (5, 4.5), **(fig_args or {})}
        _, ax = plt.subplots(**fig_args)
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

    gris = {"b": "#333333", "l": "#333333", "r": "#333333"}
    tax.boundary(linewidth=0.8, axes_colors=gris)
    tax.gridlines(multiple=grid_multiple, color="#d0d0d0", linewidth=0.5, linestyle="-")

    if size is not None:
        valores = df[size].astype(float)
        minimo, maximo = valores.min(), valores.max()
        if maximo > minimo:
            areas = np.interp(valores, (minimo, maximo), size_range)
        else:
            areas = np.full(len(valores), np.mean(size_range))
    else:
        areas = scatter_args.pop("s", 36)

    if hue is not None:
        mapa_color, puntos_color = categorical_colors(df[hue], palette)
    else:
        puntos_color = color

    tax.scatter(
        partes.values,
        s=areas,
        c=puntos_color,
        marker="o",
        alpha=alpha,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        **scatter_args,
    )

    tax.ticks(
        axis="lbr",
        multiple=tick_multiple,
        linewidth=0.6,
        tick_formats=tick_format,
        offset=0.02,
        fontsize="x-small",
        axes_colors=gris,
    )

    if corner_labels:
        tax.right_corner_label(columns[0], offset=0.04, **corner_label_args)
        tax.top_corner_label(columns[1], offset=0.2, **corner_label_args)
        tax.left_corner_label(columns[2], offset=0.04, **corner_label_args)

    if labels is not None and labels is not False:
        rotular = partes.index if labels is True else [i for i in partes.index if i in set(labels)]
        textos = []
        for idx in rotular:
            x, y = project_point(partes.loc[idx].values)
            textos.append(ax.text(x, y, str(idx), ha="center", va="center", zorder=4, **label_args))
        if avoid_collisions and textos:
            adjust_text(
                textos,
                ax=ax,
                arrowprops={"arrowstyle": "-", "color": "#808080", "lw": 0.5},
            )

    # la leyenda de tamaño va primero: categorical_color_legend conserva la
    # leyenda que encuentra en el eje y agrega la suya.
    if size is not None and size_legend:
        if size_legend_values is None:
            size_legend_values = [minimo, valores.median(), maximo]
        handles = [
            ax.scatter(
                [], [], s=np.interp(v, (minimo, maximo), size_range),
                color="#8c8c8c", edgecolor="white", linewidth=0.4,
                label=size_legend_format.format(v),
            )
            for v in size_legend_values
        ]
        size_legend_args = {
            "loc": "upper right", "fontsize": "x-small", "title": size,
            "title_fontsize": "x-small", "labelspacing": 1.2, **size_legend_args,
        }
        ax.legend(handles=handles, frameon=False, **size_legend_args)

    if hue is not None and legend:
        legend_args = {"loc": "upper left", "fontsize": "x-small", **legend_args}
        categorical_color_legend(ax, mapa_color, **legend_args)

    tax.clear_matplotlib_ticks()
    ax.set_aspect("equal")
    ax.set_axis_off()

    return tax
