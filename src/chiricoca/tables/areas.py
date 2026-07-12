import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chiricoca.base.collections import LabelCollection
from chiricoca.base.sorting import _get_sort_key


def _x_to_numeric(x):
    """Convierte el eje x a float para calculos de ventana."""
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)
    if np.issubdtype(arr.dtype, np.datetime64):
        import matplotlib.dates as mdates
        return mdates.date2num(arr)
    return np.arange(len(arr), dtype=float)


def _measure_label_bbox_data(ax, text, fontsize, fontweight, outline=False, outline_lw=2):
    """
    Mide el bbox de una etiqueta en coordenadas de datos.

    Si outline=True, aplica un path_effect Stroke con outline_lw para que el
    bbox medido refleje el tamano visual real (la etiqueta in-band se renderiza
    con outline para destacar sobre la banda).
    """
    import matplotlib.patheffects as path_effects

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    t = ax.text(0, 0, text, fontsize=fontsize, fontweight=fontweight)
    if outline:
        t.set_path_effects([
            path_effects.Stroke(linewidth=outline_lw, foreground='black'),
            path_effects.Normal(),
        ])
    bbox = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
    w = abs(bbox.x1 - bbox.x0)
    h = abs(bbox.y1 - bbox.y0)
    t.remove()
    return w, h


def _best_label_position(x_numeric, lower, upper, bbox_w, bbox_h, alpha, obstacles=None):
    """
    Encuentra el indice x donde mejor cabe una etiqueta de tamano (bbox_w, bbox_h)
    dentro de la banda definida por lower(x), upper(x), evitando solapes con
    bboxes de etiquetas previas (obstacles).

    Score = (grosor minimo en ventana de ancho bbox_w) - alpha * (rango del centro
    de la banda en esa ventana), normalizados por el grosor maximo. Posiciones que
    se solapan con obstacles reciben score = -inf.

    Parameters
    ----------
    obstacles : list of (x_left, x_right, y_bottom, y_top), opcional.

    Returns
    -------
    i_best : int
    min_height : float
        Grosor minimo conseguido en la ventana optima (0 si todos los puntos
        quedan bloqueados por obstaculos).
    """
    height = upper - lower
    center = (upper + lower) / 2.0
    n = len(x_numeric)

    if n == 0:
        return 0, 0.0

    max_height = float(height.max())
    if max_height <= 0:
        return int(np.argmax(height)), 0.0

    half_w = bbox_w / 2.0
    half_h = bbox_h / 2.0
    min_h_arr = np.empty(n)
    range_arr = np.empty(n)

    lo_ptr = 0
    hi_ptr = 0
    for i in range(n):
        while lo_ptr < n and x_numeric[i] - x_numeric[lo_ptr] > half_w:
            lo_ptr += 1
        while hi_ptr < n and x_numeric[hi_ptr] - x_numeric[i] <= half_w:
            hi_ptr += 1
        win_h = height[lo_ptr:hi_ptr]
        win_c = center[lo_ptr:hi_ptr]
        if win_h.size:
            min_h_arr[i] = win_h.min()
            range_arr[i] = win_c.max() - win_c.min()
        else:
            min_h_arr[i] = 0.0
            range_arr[i] = 0.0

    # Score combina grosor minimo (en unidades de y) menos pendiente. Al estar
    # ambos terminos en unidades de y, alpha pondera directamente: alpha=1.0
    # significa "una unidad de slope vale lo mismo que una unidad de grosor".
    score = min_h_arr - alpha * range_arr

    if obstacles:
        for i in range(n):
            label_x_left = x_numeric[i] - half_w
            label_x_right = x_numeric[i] + half_w
            label_y_bottom = center[i] - half_h
            label_y_top = center[i] + half_h
            for ox_left, ox_right, oy_bottom, oy_top in obstacles:
                if (label_x_left < ox_right and label_x_right > ox_left and
                        label_y_bottom < oy_top and label_y_top > oy_bottom):
                    score[i] = -np.inf
                    break

    i_best = int(np.argmax(score))
    if not np.isfinite(score[i_best]):
        return i_best, 0.0
    return i_best, float(min_h_arr[i_best])


def _place_callouts_outside(ax, callouts, callout_args, outline=False):
    """
    Coloca etiquetas de callout dentro del axes y resuelve solapes con adjust_text.

    Cada etiqueta empieza justo encima (o debajo) de su anchor con un pequeno
    margen, y adjust_text resuelve solapes moviendo en xy. La etiqueta solo se
    aleja del anchor lo necesario, asi los leaders son cortos cuando no hay
    competencia.

    Parameters
    ----------
    callouts : list of dict
        Cada dict tiene keys: x_anchor, y_anchor, label, side ('top' o 'bottom').
    """
    if not callouts:
        return

    from adjustText import adjust_text
    from matplotlib.transforms import blended_transform_factory

    leader_color = callout_args.get('leader_color', '#555555')
    leader_lw = callout_args.get('leader_lw', 0.5)
    fontsize = callout_args.get('fontsize', 9)
    fontweight = callout_args.get('fontweight', 'normal')
    text_color = callout_args.get('color', 'black')

    margin_frac = callout_args.get('margin_frac', 0.04)
    top_max_frac = callout_args.get('top_max_frac', 0.97)
    bot_min_frac = callout_args.get('bot_min_frac', 0.03)

    ax_ymin, ax_ymax = ax.get_ylim()
    y_range_data = max(ax_ymax - ax_ymin, 1e-12)

    text_y_transform = blended_transform_factory(ax.transData, ax.transAxes)

    for side in ('top', 'bottom'):
        group = [c for c in callouts if c['side'] == side]
        if not group:
            continue

        texts = []
        targets_x = []
        targets_y = []

        for c in group:
            # Posicion inicial: anchor + margen hacia el borde del axes.
            y_anchor_frac = (c['y_anchor'] - ax_ymin) / y_range_data
            if side == 'top':
                y_init_frac = min(top_max_frac, y_anchor_frac + margin_frac)
                y_init_frac = max(y_init_frac, y_anchor_frac + margin_frac * 0.25)
                va = 'bottom'
            else:
                y_init_frac = max(bot_min_frac, y_anchor_frac - margin_frac)
                y_init_frac = min(y_init_frac, y_anchor_frac - margin_frac * 0.25)
                va = 'top'

            t = ax.text(
                c['x_anchor'], y_init_frac, c['label'],
                ha='center', va=va,
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                transform=text_y_transform,
            )
            if outline:
                import matplotlib.patheffects as path_effects
                t.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='black'),
                    path_effects.Normal(),
                ])
            texts.append(t)
            targets_x.append(c['x_anchor'])
            targets_y.append(y_init_frac)

        if len(texts) > 1:
            default_adjust_args = dict(
                target_x=targets_x,
                target_y=targets_y,
                only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
                ensure_inside_axes=True,
                avoid_self=False,
                expand=(1.4, 1.2),
                force_text=(0.8, 0.5),
                force_pull=(0.3, 0.3),
                force_explode=(0.4, 0.2),
                iter_lim=200,
            )
            default_adjust_args.update(callout_args.get('adjust_text_args', {}))
            adjust_text(texts, ax=ax, **default_adjust_args)

        # Clamp final: la etiqueta no debe quedar sobre la banda (mas alla del
        # anchor mas el margen) ni fuera del axes.
        for t, c in zip(texts, group):
            x_final, y_final = t.get_position()
            y_anchor_frac = (c['y_anchor'] - ax_ymin) / y_range_data
            if side == 'top':
                y_min_allowed = y_anchor_frac + margin_frac * 0.25
                y_max_allowed = top_max_frac
                y_final = min(max(y_final, y_min_allowed), y_max_allowed)
            else:
                y_min_allowed = bot_min_frac
                y_max_allowed = y_anchor_frac - margin_frac * 0.25
                y_final = min(max(y_final, y_min_allowed), y_max_allowed)
            t.set_position((x_final, y_final))

            ax.annotate(
                '',
                xy=(c['x_anchor'], c['y_anchor']),
                xycoords='data',
                xytext=(x_final, y_final),
                textcoords=text_y_transform,
                arrowprops=dict(arrowstyle='-', color=leader_color, lw=leader_lw),
            )


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
    outline_labels=True,
    label_score_alpha=0.1,
    label_fit_threshold=0.7,
    label_callout_args=None,
    avoid_label_collisions=None,
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
        Umbral minimo para mostrar etiquetas.
    label_args : dict
        Argumentos para el texto de las etiquetas (fontsize, fontweight, color).
    label_rolling_window : int
        Sin uso. Se mantiene por compatibilidad.
    outline_labels : bool
        Si True, añade contorno a las etiquetas in-band.
    label_score_alpha : float
        Peso de la penalizacion por pendiente en el score de posicion. Valores
        mas altos prefieren zonas planas aunque sean mas angostas.
    label_fit_threshold : float
        Fraccion del alto del bbox de la etiqueta que debe caber dentro de la
        banda en su mejor posicion. Si el grosor minimo conseguido es menor, la
        etiqueta se saca con callout.
    label_callout_args : dict
        Configuracion del callout. Cada etiqueta empieza junto a su anchor (con
        un pequeno margen) y adjust_text resuelve solapes moviendo en xy. Keys:
        margin_frac (0.04, separacion inicial del anchor en fraccion del axes),
        top_max_frac (0.97), bot_min_frac (0.03), leader_color, leader_lw,
        color, fontsize, fontweight, side ('top', 'bottom' o None),
        adjust_text_args (override de parametros de adjustText.adjust_text).
    avoid_label_collisions : bool
        Deprecado. El nuevo placement maneja colisiones por construccion.
    label_collision_args : dict
        Deprecado. Sin efecto.
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
        Figura asociada al eje. Si None y ax es None, se crea una nueva.
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

    if label_callout_args is None:
        label_callout_args = {}

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
        if labels == 'highlight' and highlight is not None:
            if isinstance(highlight, str):
                cols_to_label = {highlight}
            else:
                cols_to_label = set(highlight)
        else:
            cols_to_label = set(df.columns)

        in_band_fontsize = label_args.get("fontsize", "medium")
        in_band_fontweight = label_args.get("fontweight", "bold")
        in_band_color = label_args.get("color", "white")

        x_numeric = _x_to_numeric(stream_x)
        in_band_collection = LabelCollection()
        callouts = []
        placed_bboxes = []

        data_y_min = float(np.min(stream_first_line))
        data_y_max = float(np.max(stream_stack[-1, :]))
        data_y_mid = (data_y_min + data_y_max) / 2.0
        callout_side_override = label_callout_args.get('side')  # 'top', 'bottom', None

        # Procesar bandas en orden de tamano descendente: las dominantes eligen
        # posicion primero y las siguientes evitan los bboxes ya colocados.
        band_candidates = []
        for i, col_name in enumerate(df.columns):
            if col_name not in cols_to_label:
                continue
            lower = stream_first_line if i == 0 else stream_stack[i - 1, :]
            upper = stream_stack[i, :]
            band_height = upper - lower
            if band_height.max() < label_threshold:
                continue
            band_candidates.append((i, col_name, lower, upper, band_height))

        band_candidates.sort(key=lambda t: -float(t[4].max()))

        for i, col_name, lower, upper, band_height in band_candidates:
            bbox_w, bbox_h = _measure_label_bbox_data(
                ax, col_name, in_band_fontsize, in_band_fontweight,
                outline=outline_labels,
            )

            i_best, min_h = _best_label_position(
                x_numeric, lower, upper, bbox_w, bbox_h, label_score_alpha,
                obstacles=placed_bboxes,
            )

            if min_h >= bbox_h * label_fit_threshold:
                x_best = stream_x[i_best]
                x_num_best = x_numeric[i_best]
                y_center = (upper[i_best] + lower[i_best]) / 2.0
                in_band_collection.add_text(col_name, x_best, y_center)
                placed_bboxes.append((
                    x_num_best - bbox_w / 2.0,
                    x_num_best + bbox_w / 2.0,
                    y_center - bbox_h / 2.0,
                    y_center + bbox_h / 2.0,
                ))
            else:
                i_peak = int(np.argmax(band_height))
                y_peak_center = (upper[i_peak] + lower[i_peak]) / 2.0
                if callout_side_override in ('top', 'bottom'):
                    side = callout_side_override
                elif baseline == 'zero':
                    side = 'top'
                else:
                    side = 'top' if y_peak_center >= data_y_mid else 'bottom'
                y_anchor = upper[i_peak] if side == 'top' else lower[i_peak]
                callouts.append({
                    'x_anchor': stream_x[i_peak],
                    'y_anchor': y_anchor,
                    'label': col_name,
                    'side': side,
                })

        in_band_collection.render(
            ax,
            fig=fig,
            color=in_band_color,
            fontweight=in_band_fontweight,
            fontsize=in_band_fontsize,
            outline=outline_labels,
        )

        if callouts:
            if outline_labels:
                callout_defaults = {
                    'fontweight': in_band_fontweight,
                    'color': in_band_color,
                }
            else:
                callout_defaults = {
                    'fontweight': 'normal',
                    'color': 'black',
                }
            merged_callout_args = {
                'fontsize': in_band_fontsize if in_band_fontsize != 'medium' else 9,
                **callout_defaults,
                **label_callout_args,
            }
            _place_callouts_outside(
                ax, callouts, merged_callout_args, outline=outline_labels,
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


def _auto_label_threshold(df, normalize):
    """Calcula automáticamente el umbral para mostrar etiquetas."""
    if normalize:
        return 0.05
    else:
        mean_area_size = df.sum(axis=0).mean()
        return mean_area_size * 0.001