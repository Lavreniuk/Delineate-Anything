import os
from osgeo import osr, ogr
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "create_geopackage_with_same_projection",
]


def create_geopackage_with_same_projection(dst_path, layer_name, projection, override_if_exists, pixel_size):
    if os.path.exists(dst_path):
        if override_if_exists:
            os.remove(dst_path)
        else:
            raise RuntimeError(f"ERROR: {dst_path} already exists!")

    proj_wkt = projection
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj_wkt)

    driver = ogr.GetDriverByName("GPKG")
    gpkg_ds = driver.CreateDataSource(dst_path)
    if gpkg_ds is None:
        raise RuntimeError(f"ERROR: Could not create GeoPackage: {dst_path}")

    gpkg_ds.SetMetadata({
        "VERTEX_STEP_X": pixel_size[0],
        "VERTEX_STEP_Y": pixel_size[1]
    })

    layer = gpkg_ds.CreateLayer(layer_name, srs=spatial_ref, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("bg", ogr.OFSTBoolean))
    layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))
    return dst_path, layer_name