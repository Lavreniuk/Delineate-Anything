import numpy as np
import pytest
from osgeo import gdal, osr

from methods.main.DataAnalyser import DataAnalyser

gdal.UseExceptions()

@pytest.mark.parametrize(
    ("nodata_value", "expected_max"),
    [(None, 65535), ([65535], 3)],
    ids=["without_nodata_value", "with_nodata_value"],
)
def test_normalization_nodata_value(tmp_path, nodata_value, expected_max):
    # Prepare test data
    path = str(tmp_path / "input.tif")
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, 10, 10, 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform((0, 10, 0, 20, 0, -10))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(32631)
    dataset.SetProjection(spatial_ref.ExportToWkt())
    values = np.full((10, 10), 65535, dtype=np.uint16)
    values[0, :3] = [1, 2, 3]
    dataset.GetRasterBand(1).WriteArray(values)
    dataset = None

    # Test
    analyser = DataAnalyser(
        [path],
        bands=[1],
        sr=None,
        norm_min=None,
        norm_max=None,
        nodata_value=nodata_value,
    )
    analyser.calcNormalizationBounds()

    if nodata_value is None:
        assert analyser.max[0] == expected_max
    else:
        assert analyser.max[0] < expected_max
    assert analyser.min[0] > 0
