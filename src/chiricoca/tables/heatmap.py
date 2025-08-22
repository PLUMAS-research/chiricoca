import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def heatmap(
    df,
    ax=None,
    fig_args=None,
    cluster_rows=False,
    cluster_cols=False,
    min_cluster_size=2,
    metric="euclidean",
    min_samples=None,
    *args,
    **kwargs
):
    """
    Heatmap con clustering opcional usando HDBSCAN

    Parameters
    ----------
    df : DataFrame
        Datos para el heatmap
    ax : matplotlib axis, optional
        Axis donde dibujar. Si None, crea uno nuevo
    fig_args : dict, optional
        Argumentos para plt.subplots()
    cluster_rows : bool, default False
        Si aplicar clustering a las filas
    cluster_cols : bool, default False
        Si aplicar clustering a las columnas
    min_cluster_size : int, default 2
        Tamaño mínimo de cluster para HDBSCAN
    metric : str, default 'euclidean'
        Métrica de distancia para HDBSCAN
    min_samples : int, optional
        Parámetro min_samples para HDBSCAN
    """

    # Si no hay clustering, usar la función original
    if not cluster_rows and not cluster_cols:
        if ax is None:
            if fig_args is None:
                fig_args = {}
            fig, ax = plt.subplots(**fig_args)

        sns.heatmap(df, ax=ax, *args, **kwargs)
        return ax

    # Clustering habilitado
    from hdbscan import HDBSCAN

    # Configurar figura con subplots para dendrogramas
    if fig_args is None:
        fig_args = {"figsize": (6, 4)}

    fig = plt.figure(**fig_args)

    # Determinar layout según qué clustering está habilitado
    if cluster_rows and cluster_cols:
        # Ambos clustering
        ax_row = plt.subplot2grid((3, 3), (1, 0), rowspan=1)
        ax_col = plt.subplot2grid((3, 3), (0, 1), colspan=1)
        ax_main = plt.subplot2grid((3, 3), (1, 1), rowspan=1, colspan=1)
    elif cluster_rows:
        # Solo filas
        ax_row = plt.subplot2grid((1, 3), (0, 0))
        ax_main = plt.subplot2grid((1, 3), (0, 1), colspan=2)
        ax_col = None
    elif cluster_cols:
        # Solo columnas
        ax_col = plt.subplot2grid((3, 1), (0, 0))
        ax_main = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
        ax_row = None

    # Parámetros para HDBSCAN
    hdbscan_kwargs = {"min_cluster_size": min_cluster_size, "metric": metric}
    if min_samples is not None:
        hdbscan_kwargs["min_samples"] = min_samples

    # Aplicar clustering y obtener orden
    df_ordered = df.copy()

    if cluster_rows:
        row_clusterer = HDBSCAN(**hdbscan_kwargs)
        row_clusterer.fit(df.values)

        row_dendro = dendrogram(
            row_clusterer.single_linkage_tree_._linkage,
            orientation="left",
            ax=ax_row,
            no_plot=False,
        )
        ax_row.set_xticks([])
        ax_row.set_yticks([])
        ax_row.set_axis_off()
        sns.despine(ax=ax_row)

        row_order = reversed(row_dendro["leaves"])
        df_ordered = df_ordered.iloc[row_order, :]

    if cluster_cols:
        col_clusterer = HDBSCAN(**hdbscan_kwargs)
        col_clusterer.fit(df.T.values)

        col_dendro = dendrogram(
            col_clusterer.single_linkage_tree_._linkage,
            orientation="top",
            ax=ax_col,
            no_plot=False,
        )
        ax_col.set_xticks([])
        ax_col.set_yticks([])
        ax_col.set_axis_off()
        sns.despine(ax=ax_col)

        col_order = col_dendro["leaves"]
        df_ordered = df_ordered.iloc[:, col_order]

    # Crear heatmap principal
    sns.heatmap(df_ordered, ax=ax_main, *args, **kwargs)

    return ax_main
