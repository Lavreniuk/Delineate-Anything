import logging

import numpy as np
import shapely
from osgeo import gdal, ogr

logger = logging.getLogger(__name__)

CELL_M = 25000.0  # side of the cells used for the local reference field size


def _area_weighted_quantile(x, q):
    if len(x) == 0:
        return 0.0
    s = np.sort(x)
    c = np.cumsum(s)
    return float(s[np.searchsorted(c, q * c[-1])])


def _straight_runs(p, min_len=200):
    """Number of exactly horizontal / vertical boundary segments >= min_len (inference tile edges)."""
    n = 0
    for r in shapely.get_rings(shapely.get_parts(p)):
        c = shapely.get_coordinates(r)
        u, v = c[:-1], c[1:]
        axis = (np.abs(u[:, 1] - v[:, 1]) < 0.01) | (np.abs(u[:, 0] - v[:, 0]) < 0.01)
        n += int(((np.hypot(*(v - u).T) >= min_len) & axis).sum())
    return n


def _axis_share(p):
    """Share of the boundary length in horizontal / vertical segments >= 20 m."""
    t = total = 0.0
    for r in shapely.get_rings(shapely.get_parts(p)):
        c = shapely.get_coordinates(r)
        u, w = c[:-1], c[1:]
        length = np.hypot(*(w - u).T)
        total += length.sum()
        d = np.abs(w - u)
        axis = (np.minimum(d[:, 0], d[:, 1]) <= 0.02 * np.maximum(length, 1e-9)) & (length >= 20)
        t += length[axis].sum()
    return t / max(total, 1)


class _ImageEdges:
    """Interior image gradients of a polygon, compared with those of normal reference fields."""

    def __init__(self, image_path, bands):
        import rasterio
        self.ds = rasterio.open(image_path)
        self.bands = bands

    def interior(self, p):
        from rasterio.features import geometry_mask
        from rasterio.windows import Window, from_bounds
        q = shapely.buffer(p, -2 * abs(self.ds.transform.a))
        if shapely.is_empty(q):
            return None
        try:
            w = from_bounds(*shapely.bounds(q), self.ds.transform).round_offsets().round_lengths()
            # pixels outside the image would be masked out anyway; boundless reads are slow on a VRT
            w = w.intersection(Window(0, 0, self.ds.width, self.ds.height))
            im = self.ds.read(self.bands, window=w).astype(float)
        except Exception:
            return None
        if im.shape[1] < 3 or im.shape[2] < 3:
            return None
        m = ~geometry_mask([q], out_shape=im.shape[1:], transform=self.ds.window_transform(w)) & (im > 0).all(0)
        if m.sum() < 30:
            return None
        return np.maximum.reduce([np.hypot(*np.gradient(im[k])) for k in range(len(self.bands))])[m]


def find_giants(geoms, tile_m, image_path=None, image_bands=(3, 4)):
    """
    Indices of "giant" polygons: fields chained together across many inference tiles over low-contrast land
    (pasture, scrub, hills, blocks of small fields), often with straight tile-edge segments. All thresholds are
    relative to the local field sizes, so real big fields (e.g. UA, KZ, US) are kept.

    geoms: shapely polygons in a projected CRS (metres); tile_m: inference tile side on the ground, metres.
    image_path: optional image the fields were delineated from; adds the "fields inside the polygon" evidence.
    """
    g = np.asarray(geoms, dtype=object)
    a = shapely.area(g)
    b = shapely.bounds(g)
    spanned = np.ceil((b[:, 2] - b[:, 0]) / tile_m + 1e-9) * np.ceil((b[:, 3] - b[:, 1]) / tile_m + 1e-9)
    clean = spanned <= 2  # reference sizes come only from polygons within one or two tiles
    ix = np.floor((b[:, 0] + b[:, 2]) / 2 / CELL_M).astype(np.int64)
    iy = np.floor((b[:, 1] + b[:, 3]) / 2 / CELL_M).astype(np.int64)

    def references(mask):
        cells = {}
        for k in np.where(mask)[0]:
            cells.setdefault((ix[k], iy[k]), []).append(a[k])
        cells = {k: np.array(v) for k, v in cells.items()}
        cache = {}

        def local(i, j):
            if (i, j) not in cache:
                v = [cells[(i + di, j + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1) if (i + di, j + dj) in cells]
                cache[(i, j)] = _area_weighted_quantile(np.concatenate(v), 0.90) if v else 0.0
            return cache[(i, j)]
        return _area_weighted_quantile(a[mask], 0.99), local

    edges, gthr, ed95 = None, None, None
    if image_path:
        edges = _ImageEdges(image_path, list(image_bands))
        med = float(np.median(a)) if len(a) else 0.0
        ref = np.where(clean & (a >= 2 * med) & (a <= 10 * med) & (a >= 2e4))[0]
        ref = ref[np.random.default_rng(0).choice(len(ref), min(300, len(ref)), replace=False)]
        rs = [r for r in (edges.interior(g[k]) for k in ref) if r is not None]
        if rs:
            gthr = float(np.percentile(np.concatenate(rs), 90))
            ed95 = float(np.percentile([float((r > gthr).mean()) for r in rs], 95))

    cache = {}

    def features(k):
        if k not in cache:
            p = g[k]
            e = np.nan
            if ed95 is not None:
                r = edges.interior(p)
                if r is not None:
                    e = float((r > gthr).mean())
            cache[k] = (_straight_runs(p), sum(shapely.get_num_interior_rings(q) for q in shapely.get_parts(p)),
                        a[k] / max(shapely.area(shapely.convex_hull(p)), 1), _axis_share(p), e)
        return cache[k]

    # tile-sized blocks with axis-aligned edges never enter the references
    square = np.zeros(len(g), bool)
    for k in np.where(a >= 0.5 * tile_m * tile_m)[0]:
        square[k] = _axis_share(g[k]) >= 0.6
    big = np.where((a >= 1e6) | square)[0]
    dropped = square.copy()
    flagged = np.zeros(0, np.int64)
    # two passes: the second recomputes the references without the suspects of the first
    for _ in range(2):
        country, local = references(clean & ~dropped)
        loc0 = np.array([local(ix[k], iy[k]) for k in big])
        pre1 = (spanned[big] >= 4) & (a[big] > 3 * np.maximum(loc0, country))
        pre2 = a[big] > 3 * loc0
        cand = big[pre1 | pre2]
        loc = loc0[pre1 | pre2]
        pre1 = pre1[pre1 | pre2]
        f = np.array([features(k) for k in cand]).reshape(-1, 5)
        runs, holes, solidity, axis, edge = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
        busy = np.nan_to_num(edge, nan=0) > 1.5 * (ed95 or 9)  # many field edges inside the polygon
        ratio = a[cand] / np.maximum(loc, 1)
        g1 = pre1 & ((runs >= 3) | busy | ((holes >= 3) & (solidity < 0.6)))
        g1 |= pre1 & (ratio > 12) & ((runs >= 2) | (solidity < 0.6))
        g2 = busy & ((holes >= 3) | (solidity < 0.75) | (runs >= 2))  # blocks of fields merged into one polygon
        g3 = (holes >= 10) & (runs >= 5)
        g4 = (spanned[cand] >= 4) & (ratio >= 6) & (axis >= 0.6)  # tile-block squares on uniform land
        g5 = (spanned[cand] >= 4) & (ratio > 15)  # far bigger than any local field
        flag = g1 | g2 | g3 | g4 | g5
        flagged = cand[flag]
        dropped[:] = square
        dropped[flagged] = True
    return flagged


def filter_giants(gpkg_path, layer_name, tile_m, image_path=None, image_bands=(3, 4)):
    """Delete giant polygons from a delineation layer in place."""
    ds = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)
    srs = layer.GetSpatialRef()
    if srs is not None and srs.IsGeographic():
        logger.warning("Giant filter skipped: it needs a projected CRS (metres).")
        return 0
    fids, geoms = [], []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        fids.append(feat.GetFID())
        geoms.append(shapely.from_wkb(bytes(geom.ExportToWkb())))
    if not geoms:
        return 0
    giants = find_giants(geoms, tile_m, image_path, image_bands)
    layer.StartTransaction()
    for k in giants:
        layer.DeleteFeature(fids[k])
    layer.CommitTransaction()
    ds = None
    area = float(shapely.area(np.asarray(geoms, dtype=object)[giants]).sum()) if len(giants) else 0.0
    logger.info(f"Giant filter: removed {len(giants)} of {len(geoms)} polygons ({area / 1e6:.1f} km2).")
    return len(giants)
