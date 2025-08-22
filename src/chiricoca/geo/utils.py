import geopandas as gpd
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
    # bounding box should be in same crs

    bounds = shapely.geometry.box(*bounding_box).buffer(buffer)

    try:
        result = geodf.assign(
            geometry=lambda g: g.map(lambda x: x.intersection(bounds))
        ).pipe(lambda x: x[x.geometry.area > 0])
    except Exception:
        # sometimes there are ill-defined intersections in polygons.
        result = geodf.assign(
            geometry=lambda g: g.buffer(0).map(lambda x: x.intersection(bounds))
        ).pipe(lambda x: x[x.geometry.area > 0])

    return result.set_crs(geodf.crs)


def clip_point_geodataframe(geodf, bounding_box, buffer=0):
    bounds = shapely.geometry.box(*bounding_box).buffer(buffer)
    return geodf[geodf.within(bounds)]


def bounding_box(geodf):
    return gpd.GeoDataFrame(
        {"geometry": [shapely.geometry.box(*geodf.total_bounds)]}, crs=geodf.crs
    )
