import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

from ..base.labels import layout_labels_vertical
from ..colors.base import categorical_color_legend, categorical_colors


def _formato_segun_rango(rango):
    """Decimales suficientes para distinguir los ticks de una columna."""
    if rango >= 100:
        return "{:,.0f}"
    if rango >= 10:
        return "{:.1f}"
    return "{:.2f}"


def parallel_coordinates(
    df,
    columns=None,
    hue=None,
    ax=None,
    fig_args=None,
    palette=None,
    color="#4c4c4c",
    alpha=0.6,
    linewidth=1.0,
    line_args=None,
    n_ticks=4,
    tick_format=None,
    tick_fontsize="x-small",
    axis_color="#333333",
    labels=None,
    label_args=None,
    legend=True,
    legend_args=None,
):
    """Coordenadas paralelas: una línea quebrada por fila, un eje vertical por columna.

    Cada columna se lleva a su propio rango (mínimo abajo, máximo arriba), así
    que las columnas no necesitan compartir unidades. Los ticks de cada eje
    muestran los valores originales.

    Parameters
    ----------
    df : DataFrame
        Una fila por observación. El índice se usa como etiqueta.
    columns : list of str, optional
        Columnas a dibujar, en el orden de los ejes. Si es None, todas las
        columnas numéricas salvo ``hue``. El orden importa: solo se comparan
        con facilidad los ejes vecinos.
    hue : str, optional
        Columna categórica que define el color de cada línea.
    palette : str, list or dict, optional
        Paleta de seaborn para ``hue``, o diccionario categoría -> color.
    color : str
        Color de las líneas cuando no hay ``hue``.
    n_ticks : int
        Ticks por eje, repartidos entre el mínimo y el máximo de la columna.
    tick_format : str, optional
        Formato de los ticks, con la sintaxis de ``str.format``. Si es None,
        los decimales dependen del rango de cada columna.
    labels : bool or iterable, optional
        ``True`` rotula todas las filas en el extremo derecho; un iterable
        rotula solo los valores del índice que contiene.
    legend : bool
        Si dibujar la leyenda de ``hue``.

    Returns
    -------
    matplotlib Axes
    """
    line_args = dict(line_args or {})
    label_args = {"fontsize": "x-small", **(label_args or {})}
    legend_args = dict(legend_args or {})

    if columns is None:
        columns = [c for c in df.columns if c != hue and pd.api.types.is_numeric_dtype(df[c])]
    columns = list(columns)
    if len(columns) < 2:
        raise ValueError("Se necesitan al menos dos columnas numéricas")

    datos = df.dropna(subset=columns)
    if len(datos) < len(df):
        print(f"  {len(df) - len(datos)} filas con valores faltantes no se dibujan")

    valores = datos[columns].astype(float)
    minimos, maximos = valores.min(), valores.max()
    rango = (maximos - minimos).replace(0, 1)
    normalizado = (valores - minimos) / rango

    if ax is None:
        fig_args = {"figsize": (6, 3), **(fig_args or {})}
        _, ax = plt.subplots(**fig_args)

    posiciones = np.arange(len(columns))
    segmentos = [np.column_stack([posiciones, fila]) for fila in normalizado.values]

    if hue is not None:
        mapa_color, colores = categorical_colors(datos[hue], palette)
    else:
        colores = color

    ax.add_collection(
        LineCollection(
            segmentos, colors=colores, linewidths=linewidth, alpha=alpha, zorder=2, **line_args
        )
    )

    for i, columna in enumerate(columns):
        formato = tick_format or _formato_segun_rango(rango[columna])
        ax.plot([i, i], [0, 1], color=axis_color, linewidth=0.8, zorder=3)
        for fraccion in np.linspace(0, 1, n_ticks):
            valor = minimos[columna] + fraccion * rango[columna]
            ax.plot([i - 0.02, i], [fraccion, fraccion], color=axis_color, linewidth=0.6, zorder=3)
            ax.text(
                i - 0.03, fraccion, formato.format(valor),
                ha="right", va="center", fontsize=tick_fontsize, color=axis_color, zorder=4,
            )

    ax.set_xlim(-0.25, len(columns) - 1 + (0.6 if labels else 0.1))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(columns)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    ax.grid(False)
    sns.despine(ax=ax, left=True, bottom=True)

    if labels is not None and labels is not False:
        rotular = list(datos.index) if labels is True else [i for i in datos.index if i in set(labels)]
        ultimo = normalizado.iloc[:, -1]
        y_originales = [ultimo.loc[idx] for idx in rotular]
        textos = [str(idx) for idx in rotular]
        y_ajustadas = layout_labels_vertical(
            ax, y_originales, textos, fontsize=label_args.get("fontsize", 8), padding=0.01
        )
        x_fin = len(columns) - 1
        for idx, y0, y1, texto in zip(rotular, y_originales, y_ajustadas, textos):
            color_etiqueta = mapa_color[datos.loc[idx, hue]] if hue is not None else color
            ax.plot([x_fin, x_fin + 0.08], [y0, y1], color="#b0b0b0", linewidth=0.5, zorder=1)
            ax.text(x_fin + 0.1, y1, texto, ha="left", va="center", color=color_etiqueta, **label_args)

    if hue is not None and legend:
        legend_args = {"loc": "upper left", "bbox_to_anchor": (1.0, 1.0), "fontsize": "x-small", **legend_args}
        categorical_color_legend(ax, mapa_color, **legend_args)

    return ax
