import geopandas as gpd
import networkx as nx
import shapely


def to_point_geodataframe(df, longitude, latitude, crs="epsg:4326", drop=False):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[longitude], df[latitude]), crs=crs
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

    # una sola consulta al índice espacial da todos los pares que se tocan;
    # los polígonos sin vecinos entran como nodos aislados y reciben color
    izquierda, derecha = gdf.sindex.query(gdf.geometry, predicate="intersects")
    pares = izquierda < derecha
    G = nx.Graph()
    G.add_nodes_from(gdf.index)
    G.add_edges_from(zip(gdf.index[izquierda[pares]], gdf.index[derecha[pares]]))

    # el coloreo voraz depende del orden en que recorre los nodos; se prueban
    # varias estrategias y queda la que usa menos colores
    estrategias = ("largest_first", "smallest_last", "DSATUR", "independent_set")
    coloring = min(
        (nx.coloring.greedy_color(G, strategy=e) for e in estrategias),
        key=lambda c: max(c.values()) + 1,
    )
    gdf["color_id"] = gdf.index.map(coloring)
    k = max(coloring.values()) + 1

    return gdf, k


def projected_centroids(geometry):
    """Centroides calculados en un CRS proyectado y devueltos al original.

    El centroide de un polígono en grados no es el del polígono en el terreno,
    así que geopandas avisa cuando se pide sobre un CRS geográfico. Esto
    proyecta a la zona UTM que corresponda, calcula ahí y vuelve.
    """
    if geometry.crs is None or not geometry.crs.is_geographic:
        return geometry.centroid
    return geometry.to_crs(geometry.estimate_utm_crs()).centroid.to_crs(geometry.crs)
