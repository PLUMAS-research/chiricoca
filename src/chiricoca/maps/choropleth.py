import numpy as np
import pandas as pd
from mapclassify import FisherJenks, Quantiles
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from chiricoca.colors import (
    BIVARIATE_BLUE_ORANGE,
    add_ranged_color_legend,
    bivariate_matrix_from_palette,
    build_palette,
    fit_tick_labels,
    format_edge,
)
from chiricoca.colors.base import _inches
from chiricoca.geo.figures import figure_from_geodataframe


def choropleth_map(
    geodf,
    column,
    k=7,
    palette=None,
    default_divergent="RdBu_r",
    default_negative="Blues_r",
    default_positive="Reds",
    palette_type="light",
    legend="colorbar",
    edgecolor="white",
    palette_center=None,
    binning="uniform",
    bins=None,
    alpha=1.0,
    linewidth=1,
    zorder=1,
    cbar_args=None,
    sqrt_transform=False,
    return_map_data=True,
    missing_color="#e0e0e0",
    missing_hatch=None,
    ax=None,
    fig_args=None,
    **kwargs,
):
    """Coropleta clasificada.

    Los polígonos sin valor en `column` se dibujan con `missing_color` (y
    `missing_hatch`, un patrón de matplotlib como `"///"`), o se omiten si es
    None. `palette_center` fija el valor en que se centra una paleta
    divergente; sin él, la paleta es divergente solo si los datos cruzan cero.
    """
    cbar_args = cbar_args or {}
    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = figure_from_geodataframe(geodf, **fig_args)

    # grid ingresa este parámetro que NO usamos
    if "color" in kwargs:
        del kwargs["color"]

    sin_dato = geodf[pd.isnull(geodf[column])]
    geodf = geodf[pd.notnull(geodf[column])].copy()
    min_value, max_value = geodf[column].min(), geodf[column].max()

    if binning in ("fisher_jenks", "quantiles"):
        if sqrt_transform:
            values = geodf[column].pipe(np.sqrt)
        else:
            values = geodf[column]

        if binning == "fisher_jenks":
            binning_method = FisherJenks(values, k=k)
        else:
            binning_method = Quantiles(values, k=k)

        if sqrt_transform:
            bins = np.insert(np.power(binning_method.bins, 2), 0, geodf[column].min())
        else:
            bins = np.insert(binning_method.bins, 0, geodf[column].min())
        geodf = geodf.assign(__bin__=binning_method.yb)
    elif binning == "uniform":
        # el último borde es el máximo: las clases son (a, b] con el mínimo
        # incluido en la primera, así la leyenda muestra los valores reales
        bins = np.linspace(min_value, max_value, num=k + 1)
        geodf = geodf.assign(
            __bin__=lambda x: pd.cut(
                x[column], bins=bins, include_lowest=True, labels=False
            ).astype(int)
        )
    elif binning == "custom":
        if bins is None:
            raise ValueError("bins are needed for custom binning")
        bins = np.array(bins)
        geodf = geodf.assign(
            __bin__=lambda x: pd.cut(
                x[column], bins=bins, include_lowest=True, labels=False
            )
        )

        if pd.isnull(geodf["__bin__"]).sum() > 0:
            raise ValueError(
                "bins do not contain all possible values",
                geodf[column].min(),
                geodf[column].max(),
            )

        geodf["__bin__"] = geodf["__bin__"].astype(int)
        k = len(bins) - 1
    else:
        raise ValueError(
            "only fisher_jenks, quantiles and uniform binning are supported"
        )

    built_palette = build_palette(
        bins, palette, palette_center, default_divergent, default_negative, default_positive, palette_type
    )

    if missing_color is not None and len(sin_dato) > 0:
        sin_dato.plot(
            ax=ax, facecolor=missing_color, hatch=missing_hatch, edgecolor=edgecolor,
            linewidth=linewidth, alpha=alpha, zorder=zorder, aspect=None,
        )

    for idx, group in geodf.groupby("__bin__"):
        group.plot(
            ax=ax,
            facecolor=built_palette[idx],
            linewidth=linewidth,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=zorder,
            aspect=None,
            **kwargs,
        )

    if legend is not None and legend is not False:
        # cada clase ocupa lo mismo en la leyenda y los bordes van compactos;
        # cbar_args puede cambiar cualquiera de los dos
        opciones = {"bin_spacing": "uniform", "tick_labels": [format_edge(b) for b in bins]}
        opciones.update(cbar_args)
        cbar_ax = add_ranged_color_legend(ax, bins, built_palette, **opciones)
    else:
        cbar_ax = None

    if return_map_data:
        return ax, {
            "axis": ax,
            "cbar_axis": cbar_ax,
            "built_palette": built_palette,
            "bins": bins,
            "binned_data": geodf["__bin__"],
        }

    return ax


def bivariate_choropleth_map(
    geodf,
    col1,
    col2,
    palette=None,
    edgecolor="white",
    k=3,
    binning="uniform",
    bins1=None,
    bins2=None,
    xlabel=None,
    ylabel=None,
    legend=True,
    cbar_ax=None,
    cbar_args=None,
    palette_args=None,
    linewidth=1,
    alpha=1,
    zorder=1,
    ax=None,
    fig_args=None,
    **kwargs,
):
    cbar_args = cbar_args or {}
    palette_args = palette_args or {}
    if ax is None:
        if fig_args is None:
            fig_args = {}
        fig, ax = figure_from_geodataframe(geodf, **fig_args)

    if binning == "uniform":
        col1_binned, col1_bins = pd.cut(geodf[col1], bins=k, labels=False, retbins=True)
        col1_binned = col1_binned.rename(f"{col1}_bin_")
        col2_binned, col2_bins = pd.cut(geodf[col2], bins=k, labels=False, retbins=True)
        col2_binned = col2_binned.rename(f"{col2}_bin_")
    elif binning == "quantiles":
        col1_binned, col1_bins = pd.qcut(geodf[col1], q=k, labels=False, retbins=True)
        col1_binned = col1_binned.rename(f"{col1}_bin_")
        col2_binned, col2_bins = pd.qcut(geodf[col2], q=k, labels=False, retbins=True)
        col2_binned = col2_binned.rename(f"{col2}_bin_")
    elif binning == "fisher_jenks":
        col1_binning = FisherJenks(geodf[col1], k=k)
        col2_binning = FisherJenks(geodf[col2], k=k)
        col1_binned = pd.Series(col1_binning.yb, name=f"{col1}_bin_", index=geodf.index)
        col2_binned = pd.Series(col2_binning.yb, name=f"{col2}_bin_", index=geodf.index)
        col1_bins = np.insert(col1_binning.bins, 0, geodf[col1].min())
        col2_bins = np.insert(col2_binning.bins, 0, geodf[col2].min())
    elif binning == "custom":
        # umbrales explicitos por eje (espejo del binning custom de
        # choropleth_map); ambos deben definir k clases
        if bins1 is None or bins2 is None:
            raise ValueError("bins1 y bins2 son necesarios para binning custom")
        if len(bins1) != len(bins2):
            raise ValueError("bins1 y bins2 deben definir la misma cantidad de clases")
        k = len(bins1) - 1
        col1_bins = np.asarray(bins1)
        col2_bins = np.asarray(bins2)
        col1_binned = pd.cut(
            geodf[col1], bins=col1_bins, include_lowest=True, labels=False
        ).rename(f"{col1}_bin_")
        col2_binned = pd.cut(
            geodf[col2], bins=col2_bins, include_lowest=True, labels=False
        ).rename(f"{col2}_bin_")

        for col, binned, bins in ((col1, col1_binned, col1_bins), (col2, col2_binned, col2_bins)):
            if binned.isnull().any():
                raise ValueError(
                    f"los bins de {col} no cubren todos los valores",
                    geodf[col].min(),
                    geodf[col].max(),
                )

        col1_binned = col1_binned.astype(int)
        col2_binned = col2_binned.astype(int)
    else:
        raise ValueError(f"{binning} binning not supported")

    # se construye despues del binning porque el binning custom deriva k de
    # los umbrales entregados. Sin paleta, con tres clases va la matriz
    # accesible del módulo; con otro k se genera desde PiYG
    if palette is None:
        palette = BIVARIATE_BLUE_ORANGE if k == 3 else "PiYG"
    bivariate_palette = bivariate_matrix_from_palette(
        palette, n_colors=k, **palette_args
    )

    binned_geodf = geodf.join(col1_binned, how="inner").join(col2_binned)[
        [f"{col1}_bin_", f"{col2}_bin_", "geometry"]
    ]

    for i in range(k):
        for j in range(k):
            color_geodf = binned_geodf[
                (binned_geodf[f"{col1}_bin_"] == i)
                & (binned_geodf[f"{col2}_bin_"] == j)
            ]
            if color_geodf.empty:
                continue

            color_geodf.plot(
                ax=ax,
                color=bivariate_palette[(j, i)],
                linewidth=linewidth,
                edgecolor=edgecolor,
                alpha=alpha,
                zorder=zorder,
                aspect=None,
                **kwargs,
            )

    # legend
    if legend:
        if cbar_ax is None:
            cbar_ax = inset_axes(
                ax,
                width=cbar_args.get("width", "12%"),
                height=cbar_args.get("height", "12%"),
                loc=cbar_args.get("location", "lower center"),
                bbox_to_anchor=cbar_args.get("bbox_to_anchor", [0.0, 0.0, 1.0, 1.0]),
                bbox_transform=cbar_args.get("bbox_transform", ax.transAxes),
            )

        cbar_ax.set_xlabel(
            xlabel if xlabel else col1, fontsize=cbar_args.get("label_size", "x-small")
        )
        cbar_ax.set_ylabel(
            ylabel if ylabel else col2, fontsize=cbar_args.get("label_size", "x-small")
        )

        etiquetas_x = [format_edge(x) for x in col1_bins]
        cbar_ax.set_xticks(
            np.arange(k + 1) - 0.5,
            labels=etiquetas_x,
            fontsize=cbar_args.get("font_size", "x-small"),
        )
        fit_tick_labels(cbar_ax, etiquetas_x, _inches(cbar_args.get("width", "12%"), ax), cbar_args.get("font_size", "x-small"))
        cbar_ax.set_yticks(
            np.arange(k + 1) - 0.5,
            labels=[format_edge(x) for x in col2_bins],
            fontsize=cbar_args.get("font_size", "x-small"),
        )

        cbar_ax.imshow(bivariate_palette, origin="lower")

    return ax, bivariate_palette, cbar_ax
