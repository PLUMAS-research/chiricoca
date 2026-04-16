import h3
import geopandas as gpd
from shapely.geometry import Point, Polygon

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
    class MockGeo:
        def __init__(self, d):
            self.d = d

        @property
        def __geo_interface__(self):
            return self.d

    bounds = list(bounds)
    bounds[0] = bounds[0] - extra_margin * (bounds[2] - bounds[0])
    bounds[2] = bounds[2] + extra_margin * (bounds[2] - bounds[0])
    bounds[1] = bounds[1] - extra_margin * (bounds[3] - bounds[1])
    bounds[3] = bounds[3] + extra_margin * (bounds[3] - bounds[1])

    cell_ids = h3.h3shape_to_cells(
        h3.geo_to_h3shape(
            MockGeo(
                {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [bounds[0], bounds[1]],
                            [bounds[2], bounds[1]],
                            [bounds[2], bounds[3]],
                            [bounds[0], bounds[3]],
                        ]
                    ],
                }
            )
        ),
        res=grid_level,
    )

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
        area = zona_estudio.to_crs(crs).unary_union
        fuera = ~gpd.GeoSeries(result[col_origen], crs=crs).within(area)
        if col_destino is not None:
            fuera = fuera | ~gpd.GeoSeries(result[col_destino], crs=crs).within(area)
        n_fuera = fuera.sum()
        if n_fuera > 0:
            print(f"  {n_fuera} registros descartados: fuera del área de estudio")
        result = result[~fuera].copy()

    hex_cols = gdf_hex[["h3_cell_id", "geometry"]]
    if "centroide" not in gdf_hex.columns:
        hex_cols = hex_cols.copy()
        hex_cols["centroide"] = hex_cols.geometry.centroid

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
