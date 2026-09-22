import matplotlib.pyplot as plt
import seaborn as sns

from chiricoca.base.weights import normalize_rows
from chiricoca.base.sorting import _get_sort_key
from chiricoca.base.labels import draw_right_legend, layout_labels_horizontal


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
    legend_args=None,
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
        lateral (solo stacked=True y horizontal=False; en otro caso cae en la
        tradicional).
    legend_args : dict
        Argumentos para la leyenda: los de `ax.legend` en la tradicional, o los
        de `draw_right_legend` (`leader`, `overflow`, `fontsize`) en la lateral.
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

    # los colores van al gráfico y no a la paleta global de seaborn, que
    # cambiaría todo lo que se dibuje después en la sesión
    colors = kwargs.pop("color", None)
    if colors is None:
        colors = list(sns.color_palette(palette, n_colors=len(df.columns)))

    func_name = "bar" if not horizontal else "barh"

    getattr(df.plot, func_name)(
        ax=ax,
        stacked=stacked,
        width=bar_width,
        edgecolor="none",
        legend=False,
        color=colors,
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
        legend_args = dict(legend_args or {})
        if legend == 'right' and not horizontal and stacked:
            # centros de los segmentos de la última barra
            ultima = df.iloc[-1].to_numpy(dtype=float)
            centros = ultima.cumsum() - ultima / 2
            draw_right_legend(
                ax, centros, df.columns, colors, len(df) - 1 + bar_width / 2, sizes=ultima, **legend_args
            )
        else:
            ax.legend(**{"bbox_to_anchor": (1.0, 0.5), "loc": "center left", "frameon": False, "reverse": True, **legend_args})

    ax.ticklabel_format(
        axis="y" if not horizontal else "x", useOffset=False, style="plain"
    )
    sns.despine(ax=ax, left=True)

    if return_df:
        return ax, df

    return ax