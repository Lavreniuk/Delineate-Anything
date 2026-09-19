import numpy as np
from osgeo import gdal, osr

from methods.main.DataAnalyser import DataAnalyser

gdal.UseExceptions()

def test_normalization_excludes_nodata_value(tmp_path):
    # Prepare test data
    path = str(tmp_path / "input.tif")
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, 2, 2, 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform((0, 10, 0, 20, 0, -10))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(32631)
    dataset.SetProjection(spatial_ref.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.array([[1, 2], [3, 65535]], dtype=np.uint16))
    dataset = None

    # Test
    analyser = DataAnalyser(
        [path], bands=[1], sr=None, norm_min=None, norm_max=None, nodata_value=[65535]
    )
    analyser.calcNormalizationBounds()

    assert analyser.max[0] < 65535
    assert analyser.min[0] > 0
