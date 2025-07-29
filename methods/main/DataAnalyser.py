from osgeo import gdal, osr
import numpy as np
import math
from tqdm import tqdm

class DataAnalyser:
    def __init__(self, tiffs, bands, sr):
        self.tiffs = tiffs
        self.bands = bands
        self.sr = sr

    def calcNormalizationBounds(self):
        def calculate_percentiles(data, percentiles=(1, 99)):
            return np.percentile(data[~np.isnan(data)], percentiles)

        BANDS = self.bands
        self.min = [[] for _ in BANDS]
        self.max = [[] for _ in BANDS]

        for file in tqdm(self.tiffs):
            ds = gdal.Open(file, gdal.GA_ReadOnly)
            
            for i in range(len(BANDS)):
                rb = ds.GetRasterBand(BANDS[i])
                if rb.DataType == gdal.GDT_Byte:
                    self.min = [0 for _ in BANDS]
                    self.max = [255 for _ in BANDS]
                    ds = None
                    return

                data = rb.ReadAsArray()
                z = data[data > 0]
                p1, p99 = calculate_percentiles(z)

                self.min[i].append(p1)
                self.max[i].append(p99)
            
            ds = None

        self.min = [np.mean(self.min[i]) for i in range(len(BANDS))]
        self.max = [np.mean(self.max[i]) for i in range(len(BANDS))]

    def isCompatible(self):
        projections = []
        pixel_sizes_x = []
        pixel_sizes_y = []
        data_types = []

        for path in self.tiffs:
            ds = gdal.Open(path)
            projections.append(ds.GetProjection())
            pixel_sizes_x.append(ds.GetGeoTransform()[1])
            pixel_sizes_y.append(ds.GetGeoTransform()[5])
            data_types.append(ds.GetRasterBand(1).DataType)
            ds = None

        if len(projections) == 0:
            return False

        self.projection = projections[0]
        self.data_type = data_types[0]
        self.pixel_size_x = pixel_sizes_x[0]
        self.pixel_size_y = pixel_sizes_y[0]

        compatible = all(p == projections[0] for p in projections) \
                and all(p == pixel_sizes_x[0] for p in pixel_sizes_x) \
                and all(p == pixel_sizes_y[0] for p in pixel_sizes_y) \
                and all(t == data_types[0] for t in data_types)

        if compatible:
            self.total_bounds = self.getTotalBounds()
            if self.sr is None:
                self.tile_size = 512 if self.get_pixel_size_meters(self.tiffs[0]) < 5 else 256
            else:
                self.tile_size = 512 // self.sr

            self.scale = 512 // self.tile_size

        return compatible


    def getBounds(self, ds):
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        minx = gt[0]
        maxx = gt[0] + cols * gt[1]
        miny = gt[3] + rows * gt[5]
        maxy = gt[3]

        return (minx, miny, maxx, maxy)

    def getTotalBounds(self):
        extents = []
        for path in self.tiffs:
            ds = gdal.Open(path)
            bounds = self.getBounds(ds)
            extents.append(bounds)

        # merge extents
        minx = min(e[0] for e in extents)
        miny = min(e[1] for e in extents)
        maxx = max(e[2] for e in extents)
        maxy = max(e[3] for e in extents)
        return minx, miny, maxx, maxy
    
    def get_pixel_size_meters(self, tiff_path):
        # Open the dataset
        ds = gdal.Open(tiff_path)
        gt = ds.GetGeoTransform()
        width_units = abs(gt[1])
        height_units = abs(gt[5])
        
        # Get center coordinates
        width = ds.RasterXSize
        height = ds.RasterYSize
        center_y = gt[3] + (width / 2) * gt[4] + (height / 2) * gt[5]

        # Load projection
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
    
        if srs.IsProjected():
            scale = srs.GetLinearUnits()  # meters per unit

            width_meters = width_units * scale
            height_meters = height_units * scale
        else:
            scale = srs.GetAngularUnits() # radians per unit
            lat_rad = center_y * scale # latitude of center in radians
            earth_radius = 6371000  # meters
            lat_m = earth_radius * scale 
            lon_m = lat_m * math.cos(lat_rad)

            width_meters = width_units * lon_m
            height_meters = height_units * lat_m

        return 0.5 * (width_meters + height_meters)
    
    def get_pixel_offset(self, ds):
        gt1 = ds.GetGeoTransform()

        pixel_width = gt1[1]
        pixel_height = gt1[5]

        offset_x = (gt1[0] - self.total_bounds[0]) / pixel_width
        offset_y = (self.total_bounds[3] - gt1[3]) / abs(pixel_height)

        return int(round(offset_x)), int(round(offset_y))