"""OpenEO vector UDF: merge fields split by chunk seams, then number them 1..N.

Run on the ``raster_to_vector`` output via
``vector_cube.apply_dimension(process=udf, dimension="geometry")``.

Seams: the raster UDF encodes the chunk of every polygon in its ID
(``band0 // 4096``). Chunk cores do not overlap, so two polygons from
different chunks can only share an edge on a seam: those pairs are merged.
Pairs that only meet at a corner point are left alone.

``field_id`` is 1..N sorted north->south, west->east; ``band0`` keeps the
lowest ID of the merged pieces.
"""

import numpy as np
from openeo.udf import FeatureCollection, UdfData


def _seam_merge(gdf):
    gdf = gdf.reset_index(drop=True)
    chunk = gdf["band0"].to_numpy().astype(np.int64) // 4096
    left, right = gdf.sindex.query(gdf.geometry, predicate="touches")
    pairs = (left < right) & (chunk[left] != chunk[right])

    parent = list(range(len(gdf)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(left[pairs], right[pairs]):
        if gdf.geometry[i].intersection(gdf.geometry[j]).length > 0:  # shared edge, not a corner
            parent[find(i)] = find(j)

    gdf["_grp"] = [find(i) for i in range(len(gdf))]
    merged = gdf.dissolve("_grp", aggfunc="min").reset_index(drop=True)
    return merged


def apply_udf_data(data: UdfData) -> UdfData:
    fc = data.get_feature_collection_list()[0]
    gdf = _seam_merge(fc.data)
    pts = gdf.representative_point()
    gdf = gdf.iloc[np.lexsort((pts.x.values, -pts.y.values))].reset_index(drop=True)
    gdf["field_id"] = np.arange(1, len(gdf) + 1)
    data.set_feature_collection_list([FeatureCollection(id=fc.id, data=gdf)])
    return data
