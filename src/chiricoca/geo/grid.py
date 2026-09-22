import warnings

import h3
import geopandas as gpd
from shapely.geometry import Polygon, box

from chiricoca.geo.utils import projected_centroids

def h3_grid_from_ids(cell_ids):
    h3grid = (
        gpd.GeoDataFrame(
            {"h3_cell_id": list(map(str, cell_ids))},
            geometry=[
                Polygon(
                    list(
                        map(lambda x: tuple(reversed(x)), h3.cell_to_boundary(cell_id))
                    )
                )
                for cell_id in cell_ids
            ],
            crs="epsg:4326",
        )
        .set_index("h3_cell_id")
    )

    return h3grid    

def h3_grid_from_bounds(bounds, extra_margin=0.0, grid_level=12, crs="epsg:4326"):
    bounds = list(bounds)
    bounds[0] = bounds[0] - extra_margin * (bounds[2] - bounds[0])
    bounds[2] = bounds[2] + extra_margin * (bounds[2] - bounds[0])
    bounds[1] = bounds[1] - extra_margin * (bounds[3] - bounds[1])
    bounds[3] = bounds[3] + extra_margin * (bounds[3] - bounds[1])

    # los bounds van en longitud y latitud; `crs` solo reproyecta la salida
    cell_ids = h3.h3shape_to_cells(h3.geo_to_h3shape(box(*bounds)), res=grid_level)

    return h3_grid_from_ids(list(map(str, cell_ids))).to_crs(crs)


def asignar_celdas_h3(
    df, gdf_hex, col_origen="origen", col_destino=None, zona_estudio=None
):
    """Asigna celdas H3 a puntos de origen (y destino) mediante spatial join.

    Parameters
    ----------
    df : GeoDataFrame
        DataFrame con columnas de geometría Point para origen y (opcionalmente) destino.
        Debe estar en el mismo CRS que ``gdf_hex``.
    gdf_hex : GeoDataFrame
        Grilla H3 con columnas ``h3_cell_id`` y ``geometry``.
    col_origen : str
        Nombre de la columna con la geometría de origen.
    col_destino : str, opcional
        Nombre de la columna con la geometría de destino. Si es None, solo
        asigna celdas al origen.
    zona_estudio : GeoDataFrame, opcional
        Si se entrega, descarta filas cuyo origen (o destino) caiga fuera
        de la unión de sus geometrías.

    Returns
    -------
    DataFrame
        Copia del DataFrame con columnas ``grid_origen`` (y ``grid_destino``
        si ``col_destino`` no es None) y sus centroides correspondientes.
        Descarta filas sin celda asignada.
    """
    crs = gdf_hex.crs
    result = df.copy()

    if zona_estudio is not None:
        area = zona_estudio.to_crs(crs).union_all()
        fuera = ~gpd.GeoSeries(result[col_origen], crs=crs).within(area)
        if col_destino is not None:
            fuera = fuera | ~gpd.GeoSeries(result[col_destino], crs=crs).within(area)
        n_fuera = fuera.sum()
        if n_fuera > 0:
            print(f"  {n_fuera} registros descartados: fuera del área de estudio")
        result = result[~fuera].copy()

    if "centroide" in gdf_hex.columns:
        hex_cols = gdf_hex[["h3_cell_id", "geometry", "centroide"]]
    else:
        hex_cols = gdf_hex[["h3_cell_id", "geometry"]].copy()
        hex_cols["centroide"] = projected_centroids(hex_cols.geometry)

    def _sjoin_columna(serie_geom, prefijo):
        puntos = gpd.GeoDataFrame(
            result[[col_origen]].iloc[:, :0],
            geometry=serie_geom.values,
            crs=crs,
        ).reset_index()
        joined = gpd.sjoin(puntos, hex_cols, how="left", predicate="within")
        result[f"grid_{prefijo}"] = joined["h3_cell_id"].values
        result[f"centroide_{prefijo}"] = joined["centroide"].values

    _sjoin_columna(result[col_origen], "origen")
    cols_drop = ["grid_origen"]

    if col_destino is not None:
        _sjoin_columna(result[col_destino], "destino")
        cols_drop.append("grid_destino")

    n_antes = len(result)
    result = result.dropna(subset=cols_drop)
    n_descartados = n_antes - len(result)
    if n_descartados > 0:
        print(f"  {n_descartados} registros descartados: sin celda H3 asignada")

    return result


def count_in_grid(points, grid_level=8, column="n", grid=None, agg=None):
    """Cuenta o agrega los points que caen en cada celda de una grilla H3.

    Resuelve en un llamado la secuencia de crear la grilla, cruzarla con los
    points y volver a unir el resultado, que de otro modo son cuatro pasos y
    obliga a conocer el nombre interno del índice de celdas.

    Parameters
    ----------
    points : GeoDataFrame
        Geometrías de punto. Se usa su bounding box si no se entrega `grid`.
    grid_level : int
        Resolución H3 de la grilla que se construye cuando no se entrega una.
    column : str
        Nombre de la columna de conteo.
    agg : dict, opcional
        Agregaciones por columna, como las de `DataFrame.agg`. Si se entrega,
        el resultado trae esas columnas además del conteo.

    Returns
    -------
    GeoDataFrame con una fila por celda con al menos un punto.
    """
    if grid is None:
        grid = h3_grid_from_bounds(points.total_bounds, grid_level=grid_level)

    indice = grid.index.name
    cruce = gpd.sjoin(points, grid.reset_index(), predicate="within")

    if len(cruce) < len(points):
        perdidos = len(points) - len(cruce)
        warnings.warn(
            f"{perdidos} de {len(points)} points quedaron fuera de la grilla "
            "y no se contaron. Suele ser el borde: subir extra_margin al crear "
            "la grilla, o entregarla ya construida.",
            stacklevel=2,
        )

    resultado = cruce.groupby(indice).size().rename(column).to_frame()
    if agg:
        resultado = resultado.join(cruce.groupby(indice).agg(agg))

    return grid.join(resultado, how="inner")
