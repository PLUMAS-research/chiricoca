import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chiricoca.base.collections import LabelCollection


def stacked_areas(ax, df, baseline="zero", color_dict=None, **kwargs):
    """
    Dibuja la visualización de la distribución cumulativa de distintas categorías a lo largo de una variable continua y obtiene
    los parámetros de la figura necesarios para generar un gráfico con la función `streamgraph`_.

    Parameters
    ----------
    ax : matplotlib.axes
        El eje en el cual se dibujará el gráfico.
    df : DataFrame
        Un DataFrame que contiene los datos para generar el gráfico.
    baseline : str, default="zero", opcional
        El método utilizado para calcular la línea base de las áreas apiladas.
    color_dict : dict, default=None, opcional
        Un diccionario que mapea los nombres de las categorías a los colores a utilizar para rellenar las áreas corresppondientes.
    **kwargs
        Argumentos adicionales que permiten personalizar el gráfico.
        Una lista completa de las opciones disponibles se encuentra en la documentación de `Matplotlib <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`__.

    Returns
    -------
    x : ndarray
        Arreglo que contiene los valores en el eje x del gráfico.
    first_line : ndarray
        Arreglo que contiene los valores de la primera línea base de las áreas apiladas.
    stack : ndarray
        Arreglo que contiene los valores de las áreas apiladas.
    """
    stack = np.cumsum(df.T.values, axis=0)
    x = df.index.values
    y = df.T.values
    m = y.shape[0]

    if baseline == "zero":
        first_line = np.zeros(x.shape)
    else:
        first_line = (y * (m - 0.5 - np.arange(m)[:, None])).sum(0)
        first_line /= -m
        first_line -= np.min(first_line)
        stack += first_line

    color = color_dict[df.columns[0]] if color_dict is not None else None
    ax.fill_between(x, first_line, stack[0, :], facecolor=color, **kwargs)

    for i in range(len(y) - 1):
        color = color_dict[df.columns[i + 1]] if color_dict is not None else None
        ax.fill_between(x, stack[i, :], stack[i + 1, :], facecolor=color, **kwargs)

    return x, first_line, stack


def streamgraph(
    df,
    baseline="wiggle",
    labels=True,
    label_threshold=None,
    label_args=None,
    fig=None,
    facecolor=None,
    edgecolor="black",
    linewidth=0.1,
    palette="husl",
    area_colors=None,
    area_args=None,
    avoid_label_collisions=False,
    outline_labels=True,
    label_collision_args=None,
    label_rolling_window=None,
    sort_areas="sum",
    normalize=False,
    ax=None,
    fig_args=None,
    legend=False,
    legend_args=None,
    legend_loc="outer",
):
    """
    Genera un gráfico **streamgraph** a partir de los datos de un dataframe.
    Este gráfico muestra el cambio de composición o distribución de distitnas categorías a lo largo del tiempo.
    Cada categoría es representada por una franja de un color que fluye por el eje horizontal, que representa una variable continua
    como por ejemplo el paso del tiempo. La altura de la franja en un punto representa la proporción relativa de esa categoría en ese momento.
    Las franjas están apiladas una encima de la otra, por lo que la altura total de las franjas en un punto indica el cumulativo de todas las categorias.

    Parameters
    ----------
    df : DataFrame
        DataFrame que contiene los datos a visualizar. Cada columna es una categoría.
    baseline : str, default="wiggle", opcional
        El método utilizado para calcular la línea base del streamgraph.
    labels : bool, default=True, opcional
        Indica si se deben mostrar etiquetas en el gráfico.
    label_threshold : float, default=None, opcional
        Umbral para posicionar las etiquetas en el gráfico. Si es None, se calcula automáticamente.
    label_args : dict, default=None, opcional
        Argumentos adicionales para personalizar las etiquetas.
    fig : Figure, default=None, opcional
        La figura en la cual se genera el gráfico. Se utiliza para el manejo de colisiones de etiquetas.
    facecolor: string, default=None, opcional
        Un color para pintar todas las áreas. Su uso anula el de palette y area_colors.
    edgecolor: string, default="black", opcional
        Un color para pintar los bordes de las áreas.
    linewidth: float, default=0.1, opcional
        El grosor de la línea borde de cada área.
    palette: string, default="husl", opcional
        Un nombre de paleta de colores para colorear cada área. Solo se utiliza si area_colors es None.
    area_colors : dict, default=None, opcional
        Un diccionario que mapea los nombres de las categorías a los colores a utilizar para rellenar las áreas corresppondientes.
    area_args : dict, default=None, opcional
        Argumentos adicionales que permiten personalizar el gráfico.
    avoid_label_collisions : bool, default=False, opcional
        Indica si se deben evitar colisiones de etiquetas en el gráfico.
    outline_labels : bool, default=True, opcional
        Indica si se deben resaltar las etiquetas mediante un contorno.
    label_collision_args : dict, default=None, opcional
        Argumentos adicionales para manejar las colisiones de etiquetas.
    sort_areas : str, default="sum", opcional
        Método para reordenar las categorías. Opciones: "sum", "max", "peak_time", "none".
    normalize : bool, default=False, opcional
        Si True, normaliza cada fila para que sume 1 y ajusta automáticamente los límites del eje y.
    ax : matplotlib.axes, default=None, opcional
        El eje en el cual se dibujará el gráfico. Si es None, se crea automáticamente.
    fig_args : dict, default=None, opcional
        Argumentos para crear la figura si ax es None.
    legend : bool, default=False, opcional
        Si True, muestra una leyenda con las categorías y sus colores.
    legend_args : dict, default=None, opcional
        Argumentos adicionales para personalizar la leyenda.
    legend_loc : str, default="upper right", opcional
        Ubicación de la leyenda. Si es "outer", se coloca fuera del área del gráfico.
    """

    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = plt.subplots(**fig_args)

    if label_args is None:
        label_args = {}

    if area_args is None:
        area_args = {}

    if legend_args is None:
        legend_args = {}

    if label_collision_args is None:
        label_collision_args = dict(
            iter_lim=25, arrowprops=dict(arrowstyle="-", color="k", lw=0.5)
        )

    df = df.fillna(0).astype(float)

    # Normalizar datos si se solicita
    if normalize:
        df = df.div(df.sum(axis=1), axis=0).fillna(0)

    # Reordenar columnas según el criterio especificado
    if sort_areas is not None and sort_areas != "none":
        if sort_areas == "sum":
            # Ordenar por suma total (de mayor a menor)
            column_order = df.sum().sort_values(ascending=False).index
        elif sort_areas == "max":
            # Ordenar por valor máximo
            column_order = df.max().sort_values(ascending=False).index
        elif sort_areas == "peak_time":
            # Ordenar por el momento donde cada categoría alcanza su máximo
            peak_times = df.idxmax()
            column_order = peak_times.sort_values().index
        else:
            column_order = df.columns

        df = df[column_order]

    # Calcular umbral automático para etiquetas si no se especifica
    if label_threshold is None:
        label_threshold = _auto_label_threshold(df, normalize)

    # Configurar parámetros de área con valores por defecto
    # Estos parámetros se pasan directamente a fill_between via **kwargs
    if edgecolor is not None and "edgecolor" not in area_args:
        area_args["edgecolor"] = edgecolor
    if linewidth is not None and "linewidth" not in area_args:
        area_args["linewidth"] = linewidth

    # Manejo mejorado de colores
    if area_colors is None:
        if palette is not None:
            colors = sns.color_palette(palette, n_colors=len(df.columns))
            area_colors = dict(zip(df.columns.values, colors))
        else:
            # Fallback a colores por defecto de matplotlib
            colors = [f"C{i}" for i in range(len(df.columns))]
            area_colors = dict(zip(df.columns.values, colors))

    stream_x, stream_first_line, stream_stack = stacked_areas(
        ax, df, color_dict=area_colors, baseline=baseline, **area_args
    )

    ax.set_xlim([df.index.min(), df.index.max()])

    # Configurar límites del eje y automáticamente
    if normalize:
        ax.set_ylim([0, 1])
    else:
        # Para datos no normalizados, usar los valores mínimo y máximo del stack
        y_min = np.min(stream_first_line)
        y_max = np.max(stream_stack[-1, :])
        margin = (y_max - y_min) * 0.05  # 5% de margen
        y_min_lim = y_min - margin if baseline != "zero" else y_min
        ax.set_ylim([y_min_lim, y_max + margin])

    if labels:
        if label_rolling_window is None:
            label_rolling_window = _auto_rolling_window(df)

        _df = df.rolling(label_rolling_window).median()

        x_to_idx = dict(zip(_df.index, range(len(df))))
        max_x = _df.idxmax().map(x_to_idx)

        label_collection = LabelCollection()

        max_idx = max_x.values[0]
        y_value = stream_stack[0, max_idx] - stream_first_line[max_idx]

        if y_value >= label_threshold:
            label_collection.add_text(
                df.columns[0],
                stream_x[max_idx],
                stream_first_line[max_idx] * 0.5 + stream_stack[0, max_idx] * 0.5,
            )

        for i in range(1, len(df.columns)):
            max_idx = max_x.values[i]
            y_value = stream_stack[i, max_idx] - stream_stack[i - 1, max_idx]

            if y_value < label_threshold:
                continue

            label_collection.add_text(
                df.columns[i],
                stream_x[max_idx],
                stream_stack[i, max_idx] * 0.5 + stream_stack[i - 1, max_idx] * 0.5,
            )

        label_collection.render(
            ax,
            fig=fig,
            color=label_args.get("color", "white"),
            fontweight=label_args.get("fontweight", "bold"),
            fontsize=label_args.get("fontsize", "medium"),
            outline=outline_labels,
            avoid_collisions=avoid_label_collisions,
            adjustment_args=label_collision_args,
        )

    # Agregar leyenda si se solicita
    if legend:
        handles = []
        labels_list = []

        # Invertir el orden para que coincida con la visualización (de abajo hacia arriba)
        for col in reversed(df.columns):
            color = area_colors.get(col, "C0")
            handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=area_args.get("edgecolor", "none"),
                    linewidth=area_args.get("linewidth", 0),
                )
            )
            labels_list.append(col)

        # Configurar argumentos por defecto para la leyenda
        legend_defaults = {"frameon": False}

        # Manejar ubicación especial 'outer'
        if legend_loc == "outer":
            legend_defaults["bbox_to_anchor"] = (1.0, 0.0, 0.1, 1.0)
            legend_defaults["loc"] = "center left"
        else:
            legend_defaults["loc"] = legend_loc

        # Aplicar argumentos personalizados
        legend_defaults.update(legend_args)

        ax.legend(handles, labels_list, **legend_defaults)

    return ax


def _auto_rolling_window(df):
    """Determina automáticamente el tamaño de la rolling window"""
    n_points = len(df)

    # Ventana base: 2-5% del total de puntos
    base_window = max(3, int(n_points * 0.03))

    # Ajustar por variabilidad: medir "rugosidad" de los datos
    variabilities = []
    for col in df.columns:
        # Calcular diferencias consecutivas normalizadas
        series = df[col]
        if series.sum() > 0:  # Evitar divisiones por cero
            diffs = np.abs(series.diff()).fillna(0)
            # Normalizar por el valor medio de la serie
            variability = diffs.mean() / (series.mean() + 1e-8)
            variabilities.append(variability)

    if variabilities:
        avg_variability = np.mean(variabilities)
        # Si hay mucha variabilidad, usar ventana más grande
        variability_factor = min(3.0, max(0.5, avg_variability * 10))
        adjusted_window = int(base_window * variability_factor)
    else:
        adjusted_window = base_window

    # Límites razonables
    min_window = 3
    max_window = min(n_points // 10, 200)  # Máximo 10% de los datos o 200 puntos

    return max(min_window, min(max_window, adjusted_window))


def _auto_label_threshold(df, normalize):
    """Calcula automáticamente el umbral para mostrar etiquetas"""
    if normalize:
        # Para datos normalizados, usar un porcentaje del total
        # Mostrar etiquetas solo para áreas que representen al menos 5% en promedio
        threshold = 0.05
    else:
        # Para datos no normalizados, calcular basado en la magnitud de los datos
        mean_area_size = df.sum(axis=0).mean()
        # Mostrar etiquetas para áreas que sean al menos 10% del tamaño promedio
        threshold = mean_area_size * 0.001

    return threshold
