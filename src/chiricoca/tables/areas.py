import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chiricoca.base.collections import LabelCollection
from chiricoca.base.sorting import _get_sort_key


def _get_sort_key_temporal(df, sort_by):
    """
    Calcula la clave de ordenamiento para columnas de un streamgraph.
    Extiende _get_sort_key con criterios temporales.
    
    Parameters
    ----------
    df : DataFrame
        Datos a ordenar. Índice es el eje temporal, columnas son categorías.
    sort_by : str o list
        Criterio de ordenamiento:
        - 'sum': suma total
        - 'max': valor máximo
        - 'name': alfabético
        - 'entropy': entropía temporal (menor = más concentrado en el tiempo)
        - 'gini': coeficiente de Gini temporal
        - 'peak_time': momento donde alcanza el máximo
        - 'first_peak': momento donde supera el 50% del máximo
        - 'onset': momento donde supera el 10% del máximo
        - str: nombre de fila específica (valor en ese momento)
        - list: orden explícito
    
    Returns
    -------
    pd.Series o None
    """
    if isinstance(sort_by, list):
        return None
    
    if sort_by == 'peak_time':
        return df.idxmax()
    
    if sort_by == 'first_peak':
        threshold = df.max() * 0.5
        def first_above(col):
            above = df.index[df[col] >= threshold[col]]
            return above[0] if len(above) > 0 else df.index[-1]
        return pd.Series({col: first_above(col) for col in df.columns})
    
    if sort_by == 'onset':
        threshold = df.max() * 0.1
        def first_above(col):
            above = df.index[df[col] >= threshold[col]]
            return above[0] if len(above) > 0 else df.index[-1]
        return pd.Series({col: first_above(col) for col in df.columns})
    
    # Para criterios no temporales, usar _get_sort_key sobre la transpuesta
    return _get_sort_key(df.T, sort_by, axis='rows')


def _build_color_dict(df, palette, highlight, highlight_palette, colors, default_color):
    """
    Construye el diccionario de colores para las áreas.
    
    Precedencia:
    1. colors (dict explícito)
    2. highlight + highlight_palette
    3. palette (o default_color para no resaltadas)
    """
    n_cols = len(df.columns)
    color_dict = {}
    
    # Normalizar highlight a lista
    if highlight is None:
        highlight_list = []
    elif isinstance(highlight, str):
        highlight_list = [highlight]
    else:
        highlight_list = list(highlight)
    
    # Generar colores base
    if highlight_list:
        # Si hay highlight, el resto va en palette base (o default_color)
        if default_color is not None:
            base_colors = [default_color] * n_cols
        elif isinstance(palette, str):
            base_colors = [sns.color_palette(palette, 1)[0]] * n_cols
        else:
            base_colors = [palette[0]] * n_cols
        
        # Colores para highlight
        n_highlight = len(highlight_list)
        if isinstance(highlight_palette, str):
            hl_colors = sns.color_palette(highlight_palette, n_colors=n_highlight)
        else:
            hl_colors = highlight_palette[:n_highlight]
        
        hl_color_map = dict(zip(highlight_list, hl_colors))
        
        for i, col in enumerate(df.columns):
            if col in hl_color_map:
                color_dict[col] = hl_color_map[col]
            else:
                color_dict[col] = base_colors[i]
    else:
        # Sin highlight, usar palette normal
        if isinstance(palette, str):
            pal_colors = sns.color_palette(palette, n_colors=n_cols)
        else:
            pal_colors = palette[:n_cols]
        color_dict = dict(zip(df.columns, pal_colors))
    
    # Override con colors explícitos
    if colors is not None:
        for col, c in colors.items():
            if col in color_dict:
                color_dict[col] = c
    
    return color_dict


def _draw_legend_right(ax, df, stream_stack, stream_first_line, color_dict, legend_args):
    """Dibuja leyenda lateral con líneas de codos."""
    from chiricoca.base.labels import layout_labels_vertical
    
    # Posiciones Y en el extremo derecho
    last_idx = -1
    positions = {}
    
    # Primera área
    y_center = (stream_first_line[last_idx] + stream_stack[0, last_idx]) / 2
    positions[df.columns[0]] = y_center
    
    # Resto de áreas
    for i in range(1, len(df.columns)):
        y_center = (stream_stack[i-1, last_idx] + stream_stack[i, last_idx]) / 2
        positions[df.columns[i]] = y_center
    
    x_right = df.index[-1]
    
    # Layout para evitar colisiones
    labels = list(df.columns)
    y_orig = [positions[col] for col in labels]
    
    fontsize = legend_args.get('fontsize', 9)
    y_final = layout_labels_vertical(ax, y_orig, labels, fontsize=fontsize)
    
    # Dibujar
    x_offset = (df.index[-1] - df.index[0]) * 0.02
    x_label = x_right + x_offset * 3
    
    for col, y_o, y_f in zip(labels, y_orig, y_final):
        color = color_dict[col]
        
        # Cuadrado de color
        ax.plot(x_label - x_offset * 0.5, y_f, 's', color=color, 
                markersize=8, clip_on=False)
        
        # Texto
        ax.text(x_label, y_f, col, va='center', fontsize=fontsize, clip_on=False)
        
        # Línea con codo si hay desplazamiento
        if abs(y_f - y_o) > 0.01:
            ax.plot([x_right, x_right + x_offset, x_label - x_offset], 
                    [y_o, y_o, y_f],
                    color='gray', lw=0.5, clip_on=False)


def stacked_areas(ax, df, baseline="zero", color_dict=None, **kwargs):
    """
    Dibuja áreas apiladas y retorna los parámetros de la figura.
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
    # Ordenamiento
    sort_areas='sum',
    sort_areas_ascending=False,
    # Colores
    palette="husl",
    highlight=None,
    highlight_palette="magma",
    colors=None,
    default_color=None,
    edgecolor="black",
    linewidth=0.1,
    # Etiquetas
    labels=True,
    label_threshold=None,
    label_args=None,
    label_rolling_window=None,
    avoid_label_collisions=False,
    outline_labels=True,
    label_collision_args=None,
    # Leyenda
    legend=False,
    legend_args=None,
    # Normalización y límites
    normalize=False,
    skip_set_xlim=False,
    skip_set_ylim=False,
    # Figura
    ax=None,
    fig=None,
    fig_args=None,
    **area_args,
):
    """
    Genera un streamgraph a partir de un dataframe.
    
    Parameters
    ----------
    df : DataFrame
        Datos a visualizar. Índice es el eje temporal, columnas son categorías.
    baseline : str, default="wiggle"
        Método para calcular la línea base: "zero" o "wiggle".
    sort_areas : str, list o False
        Ordenamiento de áreas. Opciones: 'sum', 'max', 'name', 'entropy', 'gini',
        'peak_time', 'first_peak', 'onset', nombre de fila, lista explícita, o False.
    sort_areas_ascending : bool
        Si True, ordena ascendente.
    palette : str o list
        Paleta de colores base.
    highlight : str o list
        Áreas a resaltar con highlight_palette.
    highlight_palette : str o list
        Paleta para áreas resaltadas.
    colors : dict
        Diccionario de colores explícitos (tiene precedencia).
    default_color : str
        Color para áreas no resaltadas cuando highlight está activo.
    edgecolor : str
        Color de borde de las áreas.
    linewidth : float
        Grosor del borde.
    labels : bool o 'highlight'
        True muestra todas las etiquetas, 'highlight' solo las resaltadas.
    label_threshold : float
        Umbral mínimo para mostrar etiquetas.
    label_args : dict
        Argumentos para las etiquetas.
    label_rolling_window : int
        Ventana para suavizar al calcular posición de etiquetas.
    avoid_label_collisions : bool
        Si True, ajusta posiciones para evitar colisiones.
    outline_labels : bool
        Si True, añade contorno a las etiquetas.
    label_collision_args : dict
        Argumentos para el ajuste de colisiones.
    legend : bool o 'right'
        False: sin leyenda. True: leyenda tradicional. 'right': leyenda lateral.
    legend_args : dict
        Argumentos para la leyenda.
    normalize : bool
        Si True, normaliza cada fila para que sume 1.
    skip_set_xlim : bool
        Si True, no ajusta xlim automáticamente.
    skip_set_ylim : bool
        Si True, no ajusta ylim automáticamente.
    ax : matplotlib.axes
        Eje donde dibujar. Si None, se crea uno nuevo.
    fig : Figure
        Figura (necesaria para avoid_label_collisions).
    fig_args : dict
        Argumentos para crear la figura si ax es None.
    **area_args
        Argumentos adicionales para fill_between.
    
    Returns
    -------
    ax : matplotlib.axes
    """
    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = plt.subplots(**fig_args)

    if label_args is None:
        label_args = {}

    if legend_args is None:
        legend_args = {}

    if label_collision_args is None:
        label_collision_args = dict(
            iter_lim=25, arrowprops=dict(arrowstyle="-", color="k", lw=0.5)
        )

    df = df.fillna(0).astype(float)

    # Normalizar
    if normalize:
        df = df.div(df.sum(axis=1), axis=0).fillna(0)

    # Ordenar columnas
    if sort_areas and sort_areas != 'none':
        sort_key = _get_sort_key_temporal(df, sort_areas)
        if sort_key is not None:
            column_order = sort_key.sort_values(ascending=sort_areas_ascending).index
        else:
            # sort_areas es una lista
            column_order = [c for c in sort_areas if c in df.columns]
        df = df[column_order]

    # Umbral automático para etiquetas
    if label_threshold is None:
        label_threshold = _auto_label_threshold(df, normalize)

    # Parámetros de área
    if edgecolor is not None and "edgecolor" not in area_args:
        area_args["edgecolor"] = edgecolor
    if linewidth is not None and "linewidth" not in area_args:
        area_args["linewidth"] = linewidth

    # Construir diccionario de colores
    color_dict = _build_color_dict(
        df, palette, highlight, highlight_palette, colors, default_color
    )

    # Dibujar áreas
    stream_x, stream_first_line, stream_stack = stacked_areas(
        ax, df, color_dict=color_dict, baseline=baseline, **area_args
    )

    # Límites
    if not skip_set_xlim:
        ax.set_xlim([df.index.min(), df.index.max()])

    if not skip_set_ylim:
        if normalize:
            ax.set_ylim([0, 1])
        else:
            y_min = np.min(stream_first_line)
            y_max = np.max(stream_stack[-1, :])
            margin = (y_max - y_min) * 0.05
            y_min_lim = y_min - margin if baseline != "zero" else y_min
            ax.set_ylim([y_min_lim, y_max + margin])

    # Etiquetas
    if labels:
        # Determinar qué columnas etiquetar
        if labels == 'highlight' and highlight is not None:
            if isinstance(highlight, str):
                cols_to_label = {highlight}
            else:
                cols_to_label = set(highlight)
        else:
            cols_to_label = set(df.columns)

        if label_rolling_window is None:
            label_rolling_window = _auto_rolling_window(df)

        _df = df.rolling(label_rolling_window).median()
        x_to_idx = dict(zip(_df.index, range(len(df))))
        max_x = _df.idxmax().map(x_to_idx)

        label_collection = LabelCollection()

        # Primera área
        max_idx = max_x.values[0]
        y_value = stream_stack[0, max_idx] - stream_first_line[max_idx]
        col_name = df.columns[0]

        if y_value >= label_threshold and col_name in cols_to_label:
            label_collection.add_text(
                col_name,
                stream_x[max_idx],
                stream_first_line[max_idx] * 0.5 + stream_stack[0, max_idx] * 0.5,
            )

        # Resto de áreas
        for i in range(1, len(df.columns)):
            max_idx = max_x.values[i]
            y_value = stream_stack[i, max_idx] - stream_stack[i - 1, max_idx]
            col_name = df.columns[i]

            if y_value < label_threshold or col_name not in cols_to_label:
                continue

            label_collection.add_text(
                col_name,
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

    # Leyenda
    if legend:
        if legend == 'right':
            _draw_legend_right(ax, df, stream_stack, stream_first_line, 
                             color_dict, legend_args)
        else:
            # Leyenda tradicional
            handles = []
            labels_list = []
            
            for col in reversed(df.columns):
                color = color_dict.get(col, "C0")
                handles.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                 edgecolor=area_args.get("edgecolor", "none"),
                                 linewidth=area_args.get("linewidth", 0))
                )
                labels_list.append(col)

            legend_defaults = {"frameon": False}
            legend_defaults.update(legend_args)
            ax.legend(handles, labels_list, **legend_defaults)

    return ax


def _auto_rolling_window(df):
    """Determina automáticamente el tamaño de la rolling window."""
    n_points = len(df)
    base_window = max(3, int(n_points * 0.03))

    variabilities = []
    for col in df.columns:
        series = df[col]
        if series.sum() > 0:
            diffs = np.abs(series.diff()).fillna(0)
            variability = diffs.mean() / (series.mean() + 1e-8)
            variabilities.append(variability)

    if variabilities:
        avg_variability = np.mean(variabilities)
        variability_factor = min(3.0, max(0.5, avg_variability * 10))
        adjusted_window = int(base_window * variability_factor)
    else:
        adjusted_window = base_window

    min_window = 3
    max_window = min(n_points // 10, 200)

    return max(min_window, min(max_window, adjusted_window))


def _auto_label_threshold(df, normalize):
    """Calcula automáticamente el umbral para mostrar etiquetas."""
    if normalize:
        return 0.05
    else:
        mean_area_size = df.sum(axis=0).mean()
        return mean_area_size * 0.001