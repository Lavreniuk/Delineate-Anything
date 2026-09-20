import pytest
from osgeo import gdal, ogr, osr

from methods.main.inference import postdelineation_merge
from methods.main.utils import create_geopackage_with_same_projection

gdal.UseExceptions()


def _make_layer(tmp_path, epsg):
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(epsg)

    gpkg_path, layer_name = create_geopackage_with_same_projection(
        str(tmp_path / "test.gpkg"),
        "fields",
        spatial_ref.ExportToWkt(),
        override_if_exists=True,
        pixel_size=[1.0, 1.0],
    )
    return gpkg_path, layer_name


def _add_feature(layer, wkt, field_id, bg):
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
    feat.SetField("id", field_id)
    feat.SetField("bg", bg)
    feat.SetField("area", 0.0)
    layer.CreateFeature(feat)


@pytest.mark.parametrize(
    ("epsg", "side", "gap", "max_area_m2"),
    [(32631, 10, 1e-9, 1000), (4326, 0.0001, 1e-13, 1000)],
    ids=["utm_meters", "wgs84_degrees"],
)
def test_postdelineation_merge_closes_subpixel_gap(
    tmp_path, epsg, side, gap, max_area_m2
):
    """Test if sub-pixel gaps between adjacent field fragments are closed.
    
    Adjacent polygonization worker chunks compute vertices from independent
    geotransforms, so shared edges can be off by sub-pixel float error. The size
    of that error scales with coordinate magnitude, so meters-scale (UTM) and
    degrees-scale (WGS84) coordinates warrant very different realistic gap sizes.

    `side` is the size of each fragment (roughly 11m x 11m in both cases), and
    `max_area_m2` bounds the merged area to catch the gap-closing buffer being
    applied at the wrong scale (e.g. a fixed 0.01 map-unit buffer would be ~1km
    wide, not ~1cm, in a geographic CRS).
    """
    gpkg_path, layer_name = _make_layer(tmp_path, epsg=epsg)

    gpkg = ogr.Open(gpkg_path, 1)
    layer = gpkg.GetLayerByName(layer_name)

    # Two adjacent field fragments sharing the same tile-edge id, whose common
    # boundary is offset by a sub-pixel float error between the two chunks.
    wkt1 = f"POLYGON ((0 0, {side} 0, {side} {side}, 0 {side}, 0 0))"
    _add_feature(layer, wkt1, field_id=-1, bg=0)
    wkt2 = f"POLYGON ((0 {side + gap}, {side} {side + gap}, {side} {2 * side}, 0 {2 * side}, 0 {side + gap}))"
    _add_feature(layer, wkt2, field_id=-1, bg=0)
    gpkg = None

    postdelineation_merge(
        (gpkg_path, layer_name), {"minimum_area_m2": 1, "minimum_hole_area_m2": 1}
    )

    gpkg = ogr.Open(gpkg_path, 0)
    layer = gpkg.GetLayerByName(layer_name)
    features = list(layer)

    assert len(features) == 1, "the two fragments should dissolve into a single feature"
    feature = features[0]
    assert feature.GetGeometryRef().GetGeometryName() == "POLYGON"
    area = feature.GetField("area")
    assert area < max_area_m2
