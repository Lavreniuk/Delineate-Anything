from osgeo import gdal, ogr
import numpy as np
import cv2

from affine import Affine

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

class BackgroundLoader:
    def __init__(self, background_config, lclu_path, lclu_range):
        self.lclu_path = lclu_path
        self.lclu_range = lclu_range

        self.lclu_background_classes = background_config["background_classes_from_mask"]
        if self.lclu_background_classes is not None and len(self.lclu_background_classes) == 0:
            self.lclu_background_classes = None

        self.additional_source = background_config["additional_source"]
        if self.lclu_background_classes is not None:
            self.offset = np.max(self.lclu_background_classes)
        else:
            self.offset = 1

    def get_background(self, geotransform, width, height, srs_wkt):
        gt = Affine.from_gdal(*geotransform)
        crs = rasterio.crs.CRS.from_wkt(srs_wkt)

        dst_array = None
        if self.lclu_path is not None and self.lclu_background_classes is not None:
            dst_array = BackgroundLoader.load_data_from_image(self.lclu_path, gt, width, height, crs)

            if self.lclu_range is not None:
                map = np.array([0 for _ in range(self.lclu_range)], dtype="int32")

                for val in self.lclu_background_classes:
                    map[val] = val

                dst_array = map[dst_array]
            else:
                mask = np.isin(dst_array, self.lclu_background_classes)
                dst_array = np.where(mask, dst_array, 0)
        
        if self.additional_source is not None:
            dst_vector = BackgroundLoader.load_data_from_vector(self.additional_source[0], geotransform, width, height, srs_wkt, "id")
            if dst_array is not None:
                mask = dst_vector > 0
                dst_array[mask] = self.offset + dst_vector[mask]
            else:
                dst_array = dst_vector

        return dst_array

    
    @staticmethod
    def load_data_from_image(path, geotrasform, width, height, crs):
        dst_array = np.zeros((height, width), dtype="int32")
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=geotrasform,
                dst_crs=crs,
                resampling=Resampling.nearest
            )

        return dst_array
    
    @staticmethod
    def load_data_from_vector(vector_path, geotransform, width, height, srs_wkt, attr_field=None):
        # Create in-memory target raster
        mem_drv = gdal.GetDriverByName('MEM')
        dst_ds = mem_drv.Create('', width, height, 1, gdal.GDT_Int32)
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(srs_wkt)

        # Open vector
        src_ds = ogr.Open(vector_path)
        layer = src_ds.GetLayer()

        # Burn values = feature IDs (FID) or attribute
        if attr_field is not None:
            gdal.RasterizeLayer(dst_ds, [1], layer, options=[f"ATTRIBUTE={attr_field}"])
        # else:
            # gdal.RasterizeLayer(dst_ds, [1], layer, options=["ALL_TOUCHED=TRUE"])

        arr = dst_ds.ReadAsArray()
        return np.abs(arr)