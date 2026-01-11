import h3
import geopandas as gpd
from shapely.geometry import Polygon

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
