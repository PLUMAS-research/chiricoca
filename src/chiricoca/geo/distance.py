import numpy as np
from cytoolz import valmap
from scipy.spatial.distance import pdist, squareform

from chiricoca.geo.utils import projected_centroids


def positions_from_geodataframe(geodf):
    """Diccionario índice a (x, y) de los centroides, para `networkx`."""
    return valmap(lambda x: (x.x, x.y), projected_centroids(geodf.geometry).to_dict())


def positions_to_array(geoseries):
    return np.vstack([geoseries.x.values, geoseries.y.values]).T


def calculate_distance_matrix(geodf):
    """Matriz de distancias euclidianas entre centroides.

    En un CRS proyectado las distancias van en sus unidades; en uno geográfico
    se proyecta antes a la zona UTM estimada, así que van en metros y no en
    grados.
    """
    geometria = geodf.geometry
    if geometria.crs is not None and geometria.crs.is_geographic:
        geometria = geometria.to_crs(geometria.estimate_utm_crs())
    positions = positions_to_array(geometria.centroid)
    return squareform(pdist(positions))
