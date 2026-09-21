import numpy as np
import pytest
from osgeo import gdal, osr

from methods.main.DataAnalyser import DataAnalyser

gdal.UseExceptions()

@pytest.mark.parametrize(
    ("nodata_value", "nodata_band", "expected_max", "expected_nodata_value"),
    [
        (None, None, 65535, None),
        ([65535], None, 3, [65535]),
        (65535, None, 3, [65535]),
        (1, 2, 3, 1),
    ],
    ids=["without_nodata_value", "with_nodata_value_list", "with_nodata_value_scalar", "with_nodata_band"],
)
def test_normalization_nodata_value(tmp_path, nodata_value, nodata_band, expected_max, expected_nodata_value):
    # Prepare test data
    path = str(tmp_path / "input.tif")
    driver = gdal.GetDriverByName("GTiff")
    # band 1 holds the color data; band 2 is a dedicated nodata mask, only
    # consulted by the with_nodata_band case (bands=[1] ignores it otherwise).
    dataset = driver.Create(path, 10, 10, 2, gdal.GDT_UInt16)
    dataset.SetGeoTransform((0, 10, 0, 20, 0, -10))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(32631)
    dataset.SetProjection(spatial_ref.ExportToWkt())
    values = np.full((10, 10), 65535, dtype=np.uint16)
    values[0, :3] = [1, 2, 3]
    dataset.GetRasterBand(1).WriteArray(values)
    dataset.GetRasterBand(2).WriteArray((values == 65535).astype(np.uint16))
    dataset = None

    # Test
    analyser = DataAnalyser(
        [path],
        bands=[1],
        sr=None,
        norm_min=None,
        norm_max=None,
        nodata_value=nodata_value,
        nodata_band=nodata_band,
    )

    # a scalar nodata_value is broadcast to one entry per band, so downstream
    # code can always index it without special-casing the scalar form; but a
    # dedicated nodata_band marks that single band, not one value per color
    # band, so it must stay a scalar
    assert analyser.nodata_value == expected_nodata_value

    analyser.calcNormalizationBounds()

    if nodata_value is None:
        assert analyser.max[0] == expected_max
    else:
        assert analyser.max[0] < expected_max
    assert analyser.min[0] > 0


@pytest.mark.parametrize(
    ("bands", "nodata_value", "match"),
    [
        ([3, 2, 1], [0, 65535, 65535], "nodata metadata mismatch"),
        ([1], None, "nodata_value is not configured"),
    ],
    ids=["mismatch", "config_missing"],
)
def test_warns_on_nodata_metadata_mismatch(tmp_path, bands, nodata_value, match):
    path = str(tmp_path / "input.tif")
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, 10, 10, max(bands), gdal.GDT_UInt16)
    dataset.SetGeoTransform((0, 10, 0, 20, 0, -10))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(32631)
    dataset.SetProjection(spatial_ref.ExportToWkt())

    for band_idx in range(1, max(bands) + 1):
        band = dataset.GetRasterBand(band_idx)
        band.SetNoDataValue(65535)
        band.WriteArray(np.zeros((10, 10), dtype=np.uint16))

    dataset = None

    with pytest.warns(UserWarning, match=match):
        DataAnalyser(
            [path],
            bands=bands,
            sr=None,
            norm_min=None,
            norm_max=None,
            nodata_value=nodata_value,
        )
