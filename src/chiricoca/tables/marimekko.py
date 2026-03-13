import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from chiricoca.base.labels import (
    layout_labels_horizontal,
    layout_labels_vertical,
    _draw_legend_right,
)
from chiricoca.base.sorting import _get_sort_key


def marimekko(
    df,
    width_transform=np.sqrt,
    normalize='proportions',
    sort_items=False,
    sort_items_ascending=True,
    sort_categories=False,
    sort_categories_ascending=False,
    annotate=False,
    annotate_args=None,
    xlabel_levels=False,
    legend=True,
    legend_args=None,
    palette='plasma',
    ax=None,
    fig_args=None,
    return_df=False,
):
    """
    Crea un Marimekko chart (gráfico de mosaico).
    
    El ancho de cada columna representa el total de la fila, y la altura de cada
    segmento representa la proporción de cada categoría.
    
    Parameters
    ----------
    df : DataFrame
        Datos a visualizar. El índice representa los items (eje X) y las columnas
        las categorías (segmentos apilados).
    width_transform : callable o None
        Función para transformar los anchos (ej: np.sqrt, np.log1p). None = lineal.
    normalize : bool o str
        Normalización de filas. False = sin normalizar, True o 'proportions' = 
        proporciones (0-1), 'percent' = porcentajes (0-100).
    sort_items : bool, str o list
        Ordenar filas (eje X). False = sin ordenar, True = por suma total (si no normalizado)
        o por la categoría con mayor suma (si normalizado), 'sum' = por suma total,
        'name' = alfabético, 'entropy' = por entropía (menor = más concentrado),
        'gini' = por coeficiente de Gini, str = por columna específica, list = orden explícito.
    sort_items_ascending : bool
        Dirección del ordenamiento de filas. True = menores a la izquierda.
    sort_categories : bool, str o list
        Ordenar columnas (segmentos). False = sin ordenar, True/'sum' = por suma total,
        'name' = alfabético, 'entropy' = por entropía, 'gini' = por Gini,
        str = por fila específica, list = orden explícito.
    sort_categories_ascending : bool
        Dirección del ordenamiento de columnas. False = mayores abajo.
    annotate : bool
        Mostrar porcentajes en segmentos suficientemente grandes.
    annotate_args : dict o None
        Argumentos para anotaciones:
        - min_height: fracción mínima del rango Y para anotar (default 0.04)
        - min_width: fracción mínima del ancho total para anotar (default 0.05)
        - fontsize: tamaño de fuente (default 7)
        - color: color del texto (default 'white')
    xlabel_levels : bool
        Si True, usa algoritmo de niveles para etiquetas X evitando solapamiento.
        Si False, usa ticks estándar de matplotlib.
    legend : bool
        Mostrar leyenda.
    legend_args : dict o None
        Argumentos para la leyenda. Si es 'right', usa leyenda lateral con codos.
    palette : str o list
        Paleta de colores.
    ax : matplotlib.axes.Axes o None
        Axes donde dibujar. Si None, crea una figura nueva.
    fig_args : dict o None
        Argumentos para plt.subplots si ax es None.
    return_df : bool
        Si True, retorna (ax, df_procesado).
    
    Returns
    -------
    matplotlib.axes.Axes o tuple
        El axes con el gráfico, o (ax, df) si return_df=True.
    
    Examples
    --------
    >>> ax = marimekko(df, sort_items=True, sort_categories=True, annotate=True)
    >>> ax.set_title('Distribución por categoría')
    
    >>> ax = marimekko(df, xlabel_levels=True, legend_args='right')
    >>> ax.set_xlabel('Categorías')
    """
    if ax is None:
        if fig_args is None:
            fig_args = {'figsize': (10.5, 6)}
        fig, ax = plt.subplots(**fig_args)
    
    df = df.copy()
    row_totals = df.sum(axis=1)
    
    # Normalizar primero para que el ordenamiento use proporciones
    if normalize:
        df_norm = df.div(df.sum(axis=1), axis=0)
    else:
        df_norm = df
    
    if sort_items:
        if isinstance(sort_items, list):
            df_norm = df_norm.loc[sort_items]
            row_totals = row_totals.loc[sort_items]
        else:
            # True: sum si no normalizado, por categoría más grande si normalizado
            if normalize and sort_items == True:
                largest_cat = df_norm.sum().idxmax()
                sort_key = df_norm[largest_cat]
            elif sort_items == True or sort_items == 'sum':
                sort_key = row_totals
            else:
                sort_key = _get_sort_key(df_norm, sort_items, axis='rows')
            order = sort_key.sort_values(ascending=sort_items_ascending).index
            df_norm = df_norm.loc[order]
            row_totals = row_totals.loc[order]
    
    if sort_categories:
        if isinstance(sort_categories, list):
            df_norm = df_norm[sort_categories]
        else:
            sort_key = _get_sort_key(df_norm, sort_categories, axis='columns')
            order = sort_key.sort_values(ascending=sort_categories_ascending).index
            df_norm = df_norm[order]
    
    if normalize == 'percent':
        df_norm = df_norm * 100
    
    df = df_norm
    
    if width_transform is not None:
        widths = width_transform(row_totals)
    else:
        widths = row_totals
    
    positions = np.concatenate([[0], widths.cumsum().values[:-1]])
    colors = sns.color_palette(palette, len(df.columns))
    
    # Barras apiladas
    bottoms = np.zeros(len(df))
    for color, column in zip(colors, df.columns):
        heights = df[column].values
        ax.bar(positions, heights, bottom=bottoms, width=widths.values, 
               color=color, align='edge', linewidth=0.5, edgecolor='white')
        bottoms += heights
    
    # Anotaciones en segmentos
    if annotate:
        args = {'min_height': 0.04, 'min_width': 0.05, 'fontsize': 7, 'color': 'white'}
        if annotate_args:
            args.update(annotate_args)
        
        y_range = 100 if normalize == 'percent' else 1 if normalize else bottoms.max()
        min_h = y_range * args['min_height']
        min_w = widths.sum() * args['min_width']
        fmt = '{:.0f}%' if normalize == 'percent' else '{:.0%}'
        
        bottoms_ann = np.zeros(len(df))
        for column in df.columns:
            heights = df[column].values
            for i, (pos, w, h, b) in enumerate(zip(positions, widths, heights, bottoms_ann)):
                if h > min_h and w > min_w:
                    ax.text(pos + w/2, b + h/2, fmt.format(h), 
                           ha='center', va='center',
                           fontsize=args['fontsize'], color=args['color'])
            bottoms_ann += heights
    
    # Etiquetas eje X
    x_centers = positions + widths.values / 2
    labels_x = df.index.tolist()
    
    if xlabel_levels:
        levels = layout_labels_horizontal(ax, x_centers, labels_x, fontsize=8)
        y_range = 100 if normalize == 'percent' else 1 if normalize else bottoms.max()
        tick_base = y_range * 0.04
        tick_step = y_range * 0.07
        for x, label, level in zip(x_centers, labels_x, levels):
            length = tick_base + level * tick_step
            ax.plot([x, x], [0, -length], color='black', lw=0.5, clip_on=False)
            ax.text(x, -length - y_range * 0.01, label, ha='center', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))
        ax.set_xticks([])
    else:
        ax.set_xticks(x_centers, labels_x)
    
    # Leyenda
    if legend:
        if legend_args == 'right':
            _draw_legend_right(ax, df, positions, widths, colors)
        else:
            if legend_args is None:
                legend_args = dict(bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=False)
            handles = [plt.Rectangle((0,0), 1, 1, facecolor=c) for c in reversed(colors)]
            ax.legend(handles, reversed(df.columns.tolist()), **legend_args)
    
    # Límites
    ax.set_xlim(0, positions[-1] + widths.iloc[-1])
    if normalize == 'percent':
        ax.set_ylim(0, 100)
    elif normalize:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, bottoms.max())
    
    if return_df:
        return ax, df
    return ax