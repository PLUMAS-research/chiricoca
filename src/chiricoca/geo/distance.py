import numpy as np
from cytoolz import valmap
from scipy.spatial.distance import pdist, squareform


def positions_from_geodataframe(geodf):
    """to use with networkx"""
    return valmap(lambda x: (x.x, x.y), geodf.centroid.to_dict())


def positions_to_array(geoseries):
    return np.vstack([geoseries.x.values, geoseries.y.values]).T


def calculate_distance_matrix(geodf):
    centroids = geodf.centroid
    positions = positions_to_array(centroids.geometry)
    distance_matrix = squareform(pdist(positions))
    return distance_matrix
