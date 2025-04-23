from osgeo import gdal
import numpy as np
import math

class TileLoader:
    def __init__(self, image_path, tileDim, tileStep, batchSize, skip_tiles_with_nodata, nodataValue=0, alphaBand=None, tileBands=3, dtype="uint8"):
        self.image_path = image_path
        self.tileDim = tileDim
        self.tileStep = tileStep
        self.tileBands = tileBands
        self.batchSize = batchSize
        self.alphaBand = alphaBand
        self.skip_tiles_with_nodata = skip_tiles_with_nodata
        self.nodataValue = nodataValue
        self.dtype = dtype

        ds = gdal.Open(image_path)
        self.dataSource = ds
        self.posX = 0
        self.posY = 0

        #self.estimated_batch_number = 1 + (ds.RasterXSize // self.tileStep + 1) * (ds.RasterYSize // self.tileStep + 1) // self.batchSize
        self.estimated_batch_number = math.ceil(((ds.RasterXSize + self.tileStep - 1) // self.tileStep) * ((ds.RasterYSize + self.tileStep - 1) // self.tileStep) / self.batchSize)

    def getEstimatedProgress(self):
        total_area = self.dataSource.RasterXSize * self.dataSource.RasterYSize
        covered_area = self.posY * self.dataSource.RasterXSize + self.tileDim * self.posX
        # can be (slightly) greater than 1
        return covered_area / total_area

    def nextBatch(self):
        batchCounter = 0
        dst = []
        nodata = []
        bounds = []

        ds = self.dataSource
        for posY in range(self.posY, ds.RasterYSize, self.tileStep):
            beginX = self.posX if posY == self.posY else 0

            dy = self.tileDim if (posY + self.tileDim <= ds.RasterYSize) else ds.RasterYSize - posY
            for posX in range(beginX, ds.RasterXSize, self.tileStep):
                self.posX += self.tileStep

                dx = self.tileDim if (posX + self.tileDim <= ds.RasterXSize) else ds.RasterXSize - posX
                
                if self.skip_tiles_with_nodata is not None and (self.skip_tiles_with_nodata and (dy != self.tileDim or dx != self.tileDim)):
                    continue

                alpha = None
                if self.skip_tiles_with_nodata is not None and self.alphaBand is not None:
                    band = ds.GetRasterBand(self.alphaBand)
                    alpha = band.ReadAsArray(posX, posY, dx, dy)

                    if (self.skip_tiles_with_nodata and np.any(alpha == 0)) or\
                    (not self.skip_tiles_with_nodata and np.all(alpha == 0)):
                        continue

                    nodata.append(alpha == 0)

                img = np.zeros((self.tileDim, self.tileDim, self.tileBands), dtype=self.dtype)
                for i in range(self.tileBands):
                    band = ds.GetRasterBand(i + 1)
                    img[:dy, :dx, i] = band.ReadAsArray(posX, posY, dx, dy)

                if self.skip_tiles_with_nodata is not None and alpha is None and self.nodataValue is not None:
                    if (self.skip_tiles_with_nodata and np.any(img == self.nodataValue)) or\
                    (not self.skip_tiles_with_nodata and np.all(img == self.nodataValue)):
                        continue

                    nodata.append(img[:, :, 0] == self.nodataValue)
                
                dst.append(img)
                bounds.append(tuple([posX, dx, posY, dy]))

                batchCounter += 1
                if batchCounter == self.batchSize:
                    return {
                        "images": dst,
                        "nodata": nodata,
                        "bounds": bounds
                    }

            self.posY += self.tileStep
            self.posX = 0

        if batchCounter > 0:
            return {
                "images": dst[:batchCounter],
                "nodata": nodata[:batchCounter],
                "bounds": bounds[:batchCounter]
            }
        
        else:
            return {
                "images": None,
                "nodata": None,
                "bounds": None
            }