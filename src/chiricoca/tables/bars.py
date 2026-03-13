import matplotlib.pyplot as plt
import seaborn as sns

from chiricoca.base.weights import normalize_rows
from chiricoca.base.sorting import _get_sort_key
from chiricoca.base.labels import layout_labels_horizontal, layout_labels_vertical


def _draw_legend_right_bar(ax, df, colors, stacked, normalize, bar_width):
    """Dibuja leyenda lateral con líneas y codos para barchart vertical apilado."""
    if not stacked:
        return False
    
    # Calcular posiciones Y desde la última barra
    last_row = df.iloc[-1]
    y_origins = []
    bottom = 0
    for col in df.columns:
        height = last_row[col]
        y_origins.append(bottom + height / 2)
        bottom += height
    
    # Borde derecho de la última barra
    last_x = len(df) - 1
    x_right = last_x + bar_width / 2
    
    y_max = 1 if normalize else df.sum(axis=1).max()
    
    # Normalizar posiciones para layout_labels_vertical (espera 0-1)
    if normalize:
        y_origins_norm = y_origins
    else:
        y_origins_norm = [y / y_max for y in y_origins]
    
    y_finals_norm = layout_labels_vertical(ax, y_origins_norm, df.columns.tolist(), fontsize=8)
    
    # Desnormalizar
    if normalize:
        y_finals = y_finals_norm
    else:
        y_finals = [y * y_max for y in y_finals_norm]
    
    tick_length = 0.3
    box_size = y_max * 0.02
    box_width = 0.15
    threshold = y_max * 0.01
    
    codo_indices = []
    codo_count = 0
    for y_orig, y_final in zip(y_origins, y_finals):
        if abs(y_final - y_orig) < threshold:
            codo_indices.append(None)
        else:
            codo_indices.append(codo_count)
            codo_count += 1
    
    for y_orig, y_final, label, color, codo_idx in zip(y_origins, y_finals, df.columns, colors, codo_indices):
        x_end = x_right + tick_length
        
        if codo_idx is None:
            ax.plot([x_right, x_end], [y_final, y_final], color='black', lw=0.5, clip_on=False)
        else:
            x_codo = x_right + tick_length * (0.2 + 0.15 * codo_idx)
            x_codo = min(x_codo, x_right + tick_length * 0.8)
            ax.plot([x_right, x_codo], [y_orig, y_orig], color='black', lw=0.5, clip_on=False)
            ax.plot([x_codo, x_codo], [y_orig, y_final], color='black', lw=0.5, clip_on=False)
            ax.plot([x_codo, x_end], [y_final, y_final], color='black', lw=0.5, clip_on=False)
        
        ax.add_patch(plt.Rectangle((x_end + box_width*0.3, y_final - box_size/2), 
                                   box_width, box_size,
                                   facecolor=color, edgecolor='black', lw=0.5, clip_on=False))
        
        ax.text(x_end + box_width*1.5, y_final, label, va='center', ha='left', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
    
    return True


def barchart(
    df,
    categories=None,
    palette="plasma",
    stacked=False,
    normalize=False,
    horizontal=False,
    sort_items=False,
    sort_items_ascending=False,
    sort_categories=False,
    sort_categories_ascending=False,
    fill_na_value=None,
    bar_width=0.9,
    annotate=False,
    annotate_args=None,
    xlabel_levels=False,
    legend=True,
    return_df=False,
    ax=None,
    fig_args=None,
    **kwargs
):
    """
    Crea un gráfico de barras a partir de los datos del dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Datos a visualizar. El índice se usa como eje categórico.
    categories : str, list o None
        Columnas del dataframe a graficar. Si None, usa todas las numéricas.
    palette : str o list
        Paleta de colores.
    stacked : bool
        Si True, apila las barras.
    normalize : bool
        Si True, normaliza los valores por fila.
    horizontal : bool
        Si True, barras horizontales.
    sort_items : bool, str o list
        Ordenar filas. False = sin ordenar, True = por suma total (o categoría
        más grande si normalize=True y stacked=True), 'sum' = por suma total,
        'name' = alfabético, 'entropy' = por entropía, 'gini' = por Gini,
        str = por columna específica, list = orden explícito.
    sort_items_ascending : bool
        Dirección del ordenamiento de filas.
    sort_categories : bool, str o list
        Ordenar columnas. Mismas opciones que sort_items.
    sort_categories_ascending : bool
        Dirección del ordenamiento de columnas.
    fill_na_value : objeto
        Valor para rellenar NaN.
    bar_width : float
        Ancho de las barras.
    annotate : bool
        Mostrar valores en las barras.
    annotate_args : dict
        Argumentos para bar_label.
    xlabel_levels : bool
        Si True y horizontal=False, usa etiquetado escalonado para evitar
        solapamiento en el eje X.
    legend : bool o str
        False/None = sin leyenda, True = leyenda tradicional, 'right' = leyenda
        lateral con codos (solo stacked=True y horizontal=False).
    return_df : bool
        Si True, retorna (ax, df_procesado).
    ax : matplotlib.axes.Axes
        Axes donde dibujar.
    fig_args : dict
        Argumentos para plt.subplots.
    **kwargs
        Argumentos adicionales para plot.bar/barh.

    Returns
    -------
    matplotlib.axes.Axes o tuple
    """
    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = plt.subplots(**fig_args)

    df = df.copy()

    if categories is not None:
        df = df[categories]

    if fill_na_value is not None:
        df = df.fillna(fill_na_value)

    if normalize:
        df = df.pipe(normalize_rows)

    # Ordenamiento de items (filas)
    if sort_items:
        if isinstance(sort_items, list):
            df = df.loc[sort_items]
        else:
            if normalize and stacked and sort_items == True:
                largest_cat = df.sum().idxmax()
                sort_key = df[largest_cat]
            elif sort_items == True or sort_items == 'sum':
                sort_key = df.sum(axis=1)
            else:
                sort_key = _get_sort_key(df, sort_items, axis='rows')
            if sort_key is not None:
                order = sort_key.sort_values(ascending=sort_items_ascending).index
                df = df.loc[order]

    # Ordenamiento de categorías (columnas)
    if sort_categories:
        if isinstance(sort_categories, list):
            df = df[sort_categories]
        else:
            sort_key = _get_sort_key(df, sort_categories, axis='columns')
            if sort_key is not None:
                order = sort_key.sort_values(ascending=sort_categories_ascending).index
                df = df[order]

    sns.set_palette(palette, n_colors=len(df.columns))

    func_name = "bar" if not horizontal else "barh"

    plot = getattr(df.plot, func_name)(
        ax=ax,
        stacked=stacked,
        width=bar_width,
        edgecolor="none",
        legend=False,
        **kwargs
    )

    if annotate:
        if annotate_args is None:
            annotate_args = dict()
        for container in ax.containers:
            ax.bar_label(container, **annotate_args)

    # Ajustar límites
    if not horizontal:
        ax.set_xlim(-bar_width/2, len(df) - 1 + bar_width/2)
        if normalize:
            ax.set_ylim([0, 1])
    else:
        if normalize:
            ax.set_xlim([0, 1])

    # Etiquetas escalonadas para eje X (solo vertical)
    if xlabel_levels and not horizontal:
        x_positions = range(len(df.index))
        labels = df.index.tolist()
        levels = layout_labels_horizontal(ax, x_positions, labels, fontsize=8)
        
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        tick_base = y_range * 0.02
        tick_step = y_range * 0.05
        
        for x, label, level in zip(x_positions, labels, levels):
            length = tick_base + level * tick_step
            ax.plot([x, x], [y_min, y_min - length], color='black', lw=0.5, clip_on=False)
            ax.text(x, y_min - length - y_range * 0.01, label, ha='center', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
        ax.set_xticks([])
    
    if legend:
        if legend == 'right' and not horizontal:
            colors = sns.color_palette(palette, n_colors=len(df.columns))
            if not _draw_legend_right_bar(ax, df, colors, stacked, normalize, bar_width):
                # Fallback a leyenda tradicional si no es stacked
                ax.legend(
                    bbox_to_anchor=(1.0, 0.5),
                    loc="center left",
                    frameon=False,
                    reverse=True,
                )
        else:
            ax.legend(
                bbox_to_anchor=(1.0, 0.5),
                loc="center left",
                frameon=False,
                reverse=True,
            )

    ax.ticklabel_format(
        axis="y" if not horizontal else "x", useOffset=False, style="plain"
    )
    sns.despine(ax=ax, left=True)

    if return_df:
        return ax, df

    return ax