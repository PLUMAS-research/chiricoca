import geopandas as gpd
import networkx as nx
import shapely


def to_point_geodataframe(df, longitude, latitude, crs="epsg:4326", drop=False):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=df.apply(
            lambda x: shapely.geometry.Point((x[longitude], x[latitude])), axis=1
        ),
        crs=crs,
    )

    if drop:
        gdf = gdf.drop([longitude, latitude], axis=1)

    return gdf


def clip_area_geodataframe(geodf, bounding_box, buffer=0):
    """Recorta un GeoDataFrame de polígonos a un bounding box.

    Usa clip_by_rect para mayor velocidad. Si el GeoDataFrame contiene
    una columna SHAPE (cartografía censal), la descarta.

    Parameters
    ----------
    geodf : GeoDataFrame
    bounding_box : tuple
        (xmin, ymin, xmax, ymax) en el mismo CRS que geodf.
    buffer : float
        Margen adicional alrededor del bounding box.
    """
    if buffer > 0:
        box = shapely.geometry.box(*bounding_box).buffer(buffer)
        bounding_box = box.bounds

    result = (
        geodf.assign(geometry=lambda x: x.clip_by_rect(*bounding_box))
        .set_geometry("geometry")
        .pipe(lambda x: x[~x.is_empty])
    )

    if "SHAPE" in result.columns:
        result = result.drop("SHAPE", axis=1)

    return result


def clip_point_geodataframe(geodf, bounding_box, buffer=0):
    bounds = shapely.geometry.box(*bounding_box).buffer(buffer)
    return geodf[geodf.within(bounds)]


def bounding_box(geodf):
    return gpd.GeoDataFrame(
        {"geometry": [shapely.geometry.box(*geodf.total_bounds)]}, crs=geodf.crs
    )


def k_coloreo(geodf, columna=None):
    """Asigna colores a polígonos de modo que vecinos adyacentes no compartan color.

    Disuelve el GeoDataFrame por ``columna`` (si se indica), construye un grafo
    de adyacencia y aplica coloreo greedy.

    Parameters
    ----------
    geodf : GeoDataFrame
    columna : str, opcional
        Columna por la cual disolver antes de colorear. Si es None, colorea
        cada fila individualmente.

    Returns
    -------
    geodf_coloreado : GeoDataFrame
        GeoDataFrame con columna ``color_id`` (entero desde 0).
    k : int
        Cantidad de colores utilizados.
    """
    if columna is not None:
        gdf = geodf.dissolve(by=columna)
    else:
        gdf = geodf.copy()

    gdf["geometry"] = gdf.buffer(0)

    G = nx.Graph()
    for i, geom1 in gdf.geometry.items():
        candidates = gdf.sindex.query(geom1, predicate="intersects")
        for j in candidates:
            idx_j = gdf.index[j]
            if i < idx_j:
                geom2 = gdf.geometry.iloc[j]
                if geom1.intersects(geom2) and not geom1.touches(geom2.boundary):
                    G.add_edge(i, idx_j)
                elif geom1.touches(geom2):
                    G.add_edge(i, idx_j)

    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    gdf["color_id"] = gdf.index.map(coloring)
    k = max(coloring.values()) + 1

    return gdf, k
