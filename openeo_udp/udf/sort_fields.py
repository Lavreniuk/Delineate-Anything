"""OpenEO vector UDF: merge fields split by chunk seams, then number them 1..N.

Run on the ``raster_to_vector`` output via
``vector_cube.apply_dimension(process=udf, dimension="geometry")``.

Seams: chunk cores abut, so a field cut by a seam becomes two polygons sharing
a straight edge on a line ``x (or y) = phase + k * seam_step_m``. The phase is
taken from the data: the position (mod step) carrying by far the most shared
straight-edge length. Only pairs sharing an edge on that line are merged.

``field_id`` is 1..N sorted north->south, west->east; ``band0`` keeps the
(lowest) position-based ID of the merged pieces.
"""

import numpy as np
import shapely
from openeo.udf import FeatureCollection, UdfData


def _seam_merge(gdf, step: float):
    gdf = gdf.reset_index(drop=True)
    left, right = gdf.sindex.query(gdf.geometry, predicate="touches")
    edges = []  # (i, j, axis, coord mod step, length)
    for i, j in zip(left[left < right], right[left < right]):
        shared = shapely.line_merge(gdf.geometry[i].boundary.intersection(gdf.geometry[j].boundary))
        for part in getattr(shared, "geoms", [shared]):
            if part.geom_type != "LineString":
                continue
            xy = np.asarray(part.coords)
            for axis in (0, 1):
                if (xy[:, axis] == xy[0, axis]).all():
                    edges.append((i, j, axis, round(xy[0, axis] % step, 1), part.length))

    parent = list(range(len(gdf)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for axis in (0, 1):
        totals = {}
        for _, _, ax, ph, ln in edges:
            if ax == axis:
                totals[ph] = totals.get(ph, 0.0) + ln
        ranked = sorted(totals.values(), reverse=True)
        # Only trust a phase that clearly dominates (no seam -> nothing merged).
        if not ranked or (len(ranked) > 1 and ranked[0] < 3 * ranked[1]):
            continue
        seam = max(totals, key=totals.get)
        for i, j, ax, ph, _ in edges:
            if ax == axis and ph == seam:
                parent[find(i)] = find(j)

    gdf["_grp"] = [find(i) for i in range(len(gdf))]
    merged = gdf.dissolve("_grp", aggfunc="min").reset_index(drop=True)
    return merged


def apply_udf_data(data: UdfData) -> UdfData:
    context = data.user_context or {}
    fc = data.get_feature_collection_list()[0]
    gdf = _seam_merge(fc.data, float(context.get("seam_step_m", 3840)))
    pts = gdf.representative_point()
    gdf = gdf.iloc[np.lexsort((pts.x.values, -pts.y.values))].reset_index(drop=True)
    gdf["field_id"] = np.arange(1, len(gdf) + 1)
    data.set_feature_collection_list([FeatureCollection(id=fc.id, data=gdf)])
    return data
