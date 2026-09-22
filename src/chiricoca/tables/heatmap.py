import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


def _dibujar_dendrograma(ax, valores, method, metric, orientation, color, linewidth):
    """Agrupa las filas de ``valores`` y dibuja el árbol en ``ax`` si no es None.

    Devuelve el orden de las hojas, que es el orden en que hay que poner las
    filas para que calcen con el árbol: la hoja 0 queda arriba (orientación
    ``left``) o a la izquierda (orientación ``top``), igual que la fila o
    columna 0 del heatmap de seaborn.
    """
    enlaces = linkage(valores, method=method, metric=metric)
    arbol = dendrogram(enlaces, no_plot=True)
    if ax is not None:
        n = len(arbol["leaves"])
        alto = max(max(d) for d in arbol["dcoord"])
        for hojas, alturas in zip(arbol["icoord"], arbol["dcoord"]):
            if orientation == "left":
                ax.plot(alturas, hojas, color=color, linewidth=linewidth, solid_capstyle="round")
            else:
                ax.plot(hojas, alturas, color=color, linewidth=linewidth, solid_capstyle="round")
        if orientation == "left":
            ax.set_xlim(alto * 1.05, 0)
            ax.set_ylim(10 * n, 0)
        else:
            ax.set_xlim(0, 10 * n)
            ax.set_ylim(0, alto * 1.05)
        ax.set_axis_off()
    return arbol["leaves"]


def heatmap(
    df,
    ax=None,
    fig_args=None,
    cluster_rows=False,
    cluster_cols=False,
    method="average",
    metric="euclidean",
    dendrograms=True,
    dendrogram_size=0.12,
    dendrogram_color="#555555",
    dendrogram_linewidth=0.8,
    **kwargs,
):
    """Heatmap de una matriz, con clustering jerárquico opcional de filas y columnas.

    Sin clustering es ``seaborn.heatmap`` sobre ``ax``. Con clustering la
    función crea su propia figura, reordena las filas o las columnas según el
    árbol y lo dibuja al costado; el largo de las ramas es la distancia entre
    los grupos que une.

    Parameters
    ----------
    df : DataFrame
        Matriz a dibujar. Las filas y columnas se rotulan con el índice y las
        columnas.
    ax : matplotlib Axes, optional
        Eje donde dibujar. Solo se admite sin clustering.
    fig_args : dict, optional
        Argumentos para ``plt.figure`` (con clustering) o ``plt.subplots``.
    cluster_rows, cluster_cols : bool
        Si reordenar las filas o las columnas con clustering jerárquico.
    method : str
        Criterio de enlace de ``scipy.cluster.hierarchy.linkage``: ``average``,
        ``complete``, ``single`` o ``ward``.
    metric : str
        Distancia entre filas (o columnas). Con ``ward`` tiene que ser
        ``euclidean``.
    dendrograms : bool
        Si dibujar los árboles. Con ``False`` solo se reordena.
    dendrogram_size : float
        Ancho del árbol de filas (o alto del de columnas) como fracción del
        heatmap.
    dendrogram_color, dendrogram_linewidth
        Estilo de las ramas.
    **kwargs
        Se pasan a ``seaborn.heatmap`` (``cmap``, ``robust``, ``annot``,
        ``cbar_kws``, ``linewidths``, entre otros).

    Returns
    -------
    El eje del heatmap. Con clustering, la figura se obtiene con
    ``ax.get_figure()`` y los ejes de los árboles quedan en ``fig.axes``.
    """
    if not cluster_rows and not cluster_cols:
        if ax is None:
            _, ax = plt.subplots(**(fig_args or {}))
        sns.heatmap(df, ax=ax, **kwargs)
        return ax

    if ax is not None:
        raise ValueError("Con clustering la función crea su propia figura: no se puede entregar ax")

    fig = plt.figure(**(fig_args or {}))
    ancho = [dendrogram_size, 1] if (cluster_rows and dendrograms) else [1]
    alto = [dendrogram_size, 1] if (cluster_cols and dendrograms) else [1]
    grilla = fig.add_gridspec(
        len(alto), len(ancho), width_ratios=ancho, height_ratios=alto, wspace=0.02, hspace=0.02
    )
    ax_main = fig.add_subplot(grilla[-1, -1])

    ordenada = df
    if cluster_rows:
        ax_filas = fig.add_subplot(grilla[-1, 0]) if dendrograms else None
        orden = _dibujar_dendrograma(
            ax_filas, df.values, method, metric, "left", dendrogram_color, dendrogram_linewidth
        )
        ordenada = ordenada.iloc[orden, :]
    if cluster_cols:
        ax_columnas = fig.add_subplot(grilla[0, -1]) if dendrograms else None
        orden = _dibujar_dendrograma(
            ax_columnas, df.T.values, method, metric, "top", dendrogram_color, dendrogram_linewidth
        )
        ordenada = ordenada.iloc[:, orden]

    if cluster_rows and dendrograms and kwargs.get("cbar", True) and "cbar_ax" not in kwargs:
        # los rótulos de fila van a la derecha, así que la barra de color se
        # corre más allá del rótulo más largo, en un eje propio
        from matplotlib.font_manager import FontProperties
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        puntos = FontProperties(size=plt.rcParams["ytick.labelsize"]).get_size_in_points()
        separacion = 0.15 + max(len(str(i)) for i in ordenada.index) * 0.6 * puntos / 72
        kwargs["cbar_ax"] = make_axes_locatable(ax_main).append_axes("right", size="4%", pad=separacion)

    sns.heatmap(ordenada, ax=ax_main, **kwargs)

    # la barra de color le quita ancho al heatmap; los árboles se ajustan a la
    # posición final del heatmap para que sus hojas sigan sobre las celdas
    fig.canvas.draw()
    principal = ax_main.get_position()
    if cluster_cols and dendrograms:
        arriba = ax_columnas.get_position()
        ax_columnas.set_position([principal.x0, arriba.y0, principal.width, arriba.height])
    if cluster_rows and dendrograms:
        lado = ax_filas.get_position()
        ax_filas.set_position([lado.x0, principal.y0, lado.width, principal.height])

    if cluster_rows and dendrograms:
        # las etiquetas de fila van al lado contrario del árbol, si no lo tapan.
        # seaborn las deja verticales en el lado derecho, que no revisa.
        ax_main.yaxis.tick_right()
        ax_main.yaxis.set_label_position("right")
        for etiqueta in ax_main.get_yticklabels():
            etiqueta.set_rotation(0)
    if cluster_cols and dendrograms:
        ax_main.xaxis.tick_bottom()

    return ax_main
