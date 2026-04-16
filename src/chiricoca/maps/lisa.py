import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen

from chiricoca.colors import categorical_color_legend


# Paleta por defecto para clusters LISA (basada en BrBG)
LISA_COLORS = {
    "HH": "#8c510a",
    "LL": "#01665e",
    "LH": "#d8b365",
    "HL": "#5ab4ac",
    "Not significant": "#f5f5f5",
}


def compute_lisa(geodf, column, w=None, alpha=0.05, ns_label="Not significant"):
    """Compute global Moran's I and local indicators (LISA).

    Builds Queen weights (unless ``w`` is provided), computes Moran's I and
    local Moran statistics. Returns a copy of the GeoDataFrame with cluster
    and significance columns appended.

    Parameters
    ----------
    geodf : GeoDataFrame
        GeoDataFrame with the analysis variable and geometries.
    column : str
        Name of the numeric column.
    w : libpysal.weights.W, optional
        Spatial weights matrix. If None, Queen contiguity weights are built
        from ``geodf`` and row-standardized.
    alpha : float
        Significance threshold for LISA clusters.
    ns_label : str
        Label for non-significant observations.

    Returns
    -------
    result : GeoDataFrame
        Copy of ``geodf`` with columns ``lisa_cluster``, ``lisa_pvalue``
        and ``lisa_sig``. Non-significant observations are labeled with
        ``ns_label``.
    moran_global : esda.Moran
        Global Moran's I result. Key attributes: ``I``, ``EI``,
        ``p_norm``, ``z_norm``.
    """
    if w is None:
        w = Queen.from_dataframe(geodf)
        w.transform = "r"

    moran_global = Moran(geodf[column], w)
    lisa = Moran_Local(geodf[column], w)

    result = geodf.copy()
    label_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    result["lisa_cluster"] = pd.Series(lisa.q, index=geodf.index).map(label_map)
    result["lisa_pvalue"] = lisa.p_sim
    result["lisa_sig"] = result["lisa_pvalue"] < alpha
    result.loc[~result["lisa_sig"], "lisa_cluster"] = ns_label

    return result, moran_global


def lisa_map(geodf, ax, title=None, context=None, colors=None, ns_label="Not significant"):
    """Draw a LISA cluster map with optional context boundaries.

    Parameters
    ----------
    geodf : GeoDataFrame
        GeoDataFrame with a ``lisa_cluster`` column (output of ``compute_lisa``).
    ax : matplotlib Axes
        Axes on which to draw.
    title : str, optional
        Panel title.
    context : GeoDataFrame, optional
        Polygons drawn as boundary lines over the LISA map (e.g. commune borders).
    colors : dict, optional
        Mapping ``{cluster_label: color}``. Defaults to ``LISA_COLORS``.
    ns_label : str
        Label used for non-significant observations.
    """
    if colors is None:
        colors = LISA_COLORS

    ns = geodf[geodf["lisa_cluster"] == ns_label]
    sig = geodf[geodf["lisa_cluster"] != ns_label]

    if len(ns) > 0:
        ns.plot(
            ax=ax, color=colors.get(ns_label, "#f5f5f5"),
            edgecolor="white", linewidth=0.2, zorder=1,
        )
    for cluster, color in colors.items():
        if cluster == ns_label:
            continue
        subset = sig[sig["lisa_cluster"] == cluster]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor="white", linewidth=0.3, zorder=2)

    if context is not None:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        context.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
