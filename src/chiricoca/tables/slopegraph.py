from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ..base.labels import layout_labels_vertical


def slopegraph(
    df,
    ax=None,
    colors=None,
    value_format="{:.0%}",
    labels=True,
    label_fontsize=7,
    marker_size=30,
    linewidth=1.2,
    alpha=0.8,
    connector_color="#cccccc",
    reference_line=None,
    fig_args=None,
):
    """Slopegraph de dos columnas.

    Cada fila del DataFrame se representa como una línea que conecta el valor
    de la primera columna (izquierda) con el de la segunda (derecha). Las
    etiquetas del índice se muestran a la izquierda junto con el valor; a la
    derecha solo se muestra el valor. Cuando varias filas tienen el mismo
    valor formateado en el lado derecho, se muestra una sola etiqueta con
    conectores hacia todos los puntos correspondientes.

    Parameters
    ----------
    df : DataFrame
        Exactamente dos columnas numéricas. El índice contiene las etiquetas
        de cada línea.
    ax : matplotlib.axes.Axes, optional
        Axes donde dibujar. Si es None se crea una figura nueva.
    colors : tuple of 2 colors, dict, or single color
        - tuple ``(c_right_higher, c_left_higher)``: colorea según dirección.
        - dict ``{label: color}``: color explícito por etiqueta.
        - str: color único para todas las líneas.
    value_format : str
        Formato aplicado a los valores numéricos en las etiquetas.
    labels : bool
        Si True, muestra etiquetas a ambos lados con líneas conectoras.
    label_fontsize : int
        Tamaño de fuente de las etiquetas.
    marker_size : float
        Tamaño de los marcadores en los extremos de cada línea.
    linewidth : float
    alpha : float
    connector_color : str
        Color de las líneas conectoras entre datos y etiquetas.
    reference_line : float, optional
        Valor Y donde dibujar una línea horizontal de referencia.
    fig_args : dict, optional
        Argumentos adicionales para ``plt.subplots`` (solo si ``ax`` es None).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if df.shape[1] != 2:
        raise ValueError(
            f"df debe tener exactamente 2 columnas, tiene {df.shape[1]}"
        )

    col_left, col_right = df.columns

    if fig_args is None:
        fig_args = {}

    if ax is None:
        default_height = len(df) * 0.55 + 1
        figsize = fig_args.pop("figsize", (7, default_height))
        _, ax = plt.subplots(figsize=figsize, **fig_args)

    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="#abacab",
            linewidth=0.8,
            linestyle="dotted",
            zorder=0,
        )

    if colors is None:
        colors = ("#e07b54", "#636363")

    left_vals = df[col_left].tolist()
    right_vals = df[col_right].tolist()

    # -- Resolver color por fila --
    row_colors = []
    for label, row in df.iterrows():
        lv, rv = row.iloc[0], row.iloc[1]
        if isinstance(colors, dict):
            row_colors.append(colors.get(label, "#636363"))
        elif isinstance(colors, tuple):
            row_colors.append(colors[0] if rv > lv else colors[1])
        else:
            row_colors.append(colors)

    # -- Dibujar líneas y marcadores --
    for idx, (label, row) in enumerate(df.iterrows()):
        lv, rv = row.iloc[0], row.iloc[1]
        c = row_colors[idx]
        ax.plot([0, 1], [lv, rv], color=c, linewidth=linewidth, alpha=alpha)
        ax.scatter([0, 1], [lv, rv], color=c, s=marker_size, zorder=3)

    # -- Etiquetas --
    if not labels:
        _format_axes(ax, col_left, col_right)
        return ax

    # Preparar ylim antes de layout (se necesita para los bounds)
    y_margin = (max(left_vals + right_vals) - min(left_vals + right_vals)) * 0.05
    y_lo = min(left_vals + right_vals) - y_margin
    y_hi = max(left_vals + right_vals) + y_margin
    ax.set_ylim(y_lo, y_hi)

    # --- Lado izquierdo: una etiqueta por fila (nombre + valor) ---
    left_texts = [
        f"{label}  {value_format.format(v)}"
        for label, v in zip(df.index, left_vals)
    ]
    left_nudged = layout_labels_vertical(
        ax, left_vals, left_texts, fontsize=label_fontsize,
    )

    for idx in range(len(df)):
        ax.plot(
            [0, -0.02], [left_vals[idx], left_nudged[idx]],
            color=connector_color, linewidth=0.5,
        )
        ax.text(
            -0.03, left_nudged[idx], left_texts[idx],
            ha="right", va="center", fontsize=label_fontsize,
        )

    # --- Lado derecho: agrupar etiquetas con el mismo texto formateado ---
    right_texts_per_row = [value_format.format(v) for v in right_vals]

    # Agrupar índices por texto
    groups = defaultdict(list)
    for idx, txt in enumerate(right_texts_per_row):
        groups[txt].append(idx)

    # Posición natural de cada grupo: promedio de los valores
    unique_texts = []
    unique_positions = []
    group_order = []
    for txt, indices in groups.items():
        unique_texts.append(txt)
        unique_positions.append(np.mean([right_vals[i] for i in indices]))
        group_order.append(indices)

    right_label_nudged = layout_labels_vertical(
        ax, unique_positions, unique_texts, fontsize=label_fontsize,
    )

    # Dibujar conectores y etiquetas (una por grupo)
    for g_idx in range(len(unique_texts)):
        label_y = right_label_nudged[g_idx]
        indices = group_order[g_idx]

        for row_idx in indices:
            ax.plot(
                [1, 1.02], [right_vals[row_idx], label_y],
                color=connector_color, linewidth=0.5,
            )

        ax.text(
            1.03, label_y, unique_texts[g_idx],
            ha="left", va="center", fontsize=label_fontsize,
        )

    _format_axes(ax, col_left, col_right)
    return ax


def _format_axes(ax, col_left, col_right):
    ax.set_xticks([0, 1])
    ax.set_xticklabels([col_left, col_right])
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
