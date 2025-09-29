from osgeo import gdal
import numpy as np
import cv2

class DataLoaderCached:
    def __init__(self, plan, config, batch_size, lclu_path, lclu_config):
        self.skip = config["skip"]
        self.bands = config["bands"]
        self.nodata_band = config["nodata_band"]
        self.nodata_value = config["nodata_value"]
        self.min = config["min"]
        self.max = config["max"]

        self.lclu_range = lclu_config["range"]
        self.lclu_clip_values = lclu_config["clip_classes"]
        self.lclu_filter_values = lclu_config["filter_classes"]

        self.init_plan = plan
        self.plan = plan

        self.batch_size = batch_size

        self.__load_image()
        self.__load_lclu(lclu_path)

        ds = None
        self.pos = self.plan["infile_begin"].copy()
        self.offset = self.plan["infile_begin"].copy()

    def is_compatible(self, plan):
        if self.skip:
            return True

        if self.init_plan["file"] != plan["file"]:
            return False
        
        if self.init_plan["infile_begin"][0] > plan["infile_begin"][0] or self.init_plan["infile_begin"][1] > plan["infile_begin"][1]:
            return False
        
        if self.init_plan["infile_end"][0] != plan["infile_end"][0] or self.init_plan["infile_end"][1] != plan["infile_end"][1]:
            return False

        return True
    
    def set_plan(self, plan):
        self.plan = plan
        self.pos = self.plan["infile_begin"].copy()

    def get_batch(self):
        if self.skip:
            return None, None, None

        batchCounter = 0
        batch_images = []
        batch_nodata = []
        batch_bounds = []

        pos = self.pos
        tile_size = self.plan["infile_size"]
        tile_step = self.plan["infile_step"]

        infile_begin = self.plan["infile_begin"]
        infile_end = self.plan["infile_end"]

        inregion_begin = self.plan["inregion_begin"]
        global_begin = self.plan["global_begin"]
        scale = self.plan["scale"]

        posX, posY = self.pos
        while pos[1] < infile_end[1]:
            posY = pos[1]
            while self.pos[0] < infile_end[0]:
                posX = pos[0]
                pos[0] += tile_step[0]

                image = np.zeros((tile_size[1], tile_size[0], len(self.bands)), dtype="uint8")
                nodata = np.ones((tile_size[1], tile_size[0]), dtype="uint8")

                begin_x = posX - self.offset[0]
                end_x = begin_x + tile_size[0]
                begin_y = posY - self.offset[1]
                end_y = begin_y + tile_size[1]

                nodata = self.image_cache[begin_y:end_y, begin_x:end_x, -2:]
                if np.all(nodata[:, :, 0] == 0):
                    continue

                image = self.image_cache[begin_y:end_y, begin_x:end_x, :-2]
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
                
                composite_nodata = np.empty((2, 512, 512), dtype="uint8")
                composite_nodata[0, :, :] = cv2.resize(nodata[:, :, -2], (512, 512), interpolation=cv2.INTER_NEAREST_EXACT)
                composite_nodata[1, :, :] = cv2.resize(nodata[:, :, -1], (512, 512), interpolation=cv2.INTER_NEAREST_EXACT)

                bounds = {
                    "global": tuple([global_begin[0] + scale * (posX - infile_begin[0]), global_begin[1] + scale * (posY - infile_begin[1]), scale * tile_size[0], scale * tile_size[1]]),
                    "inregion": tuple([inregion_begin[0] + scale * (posX - infile_begin[0]), inregion_begin[1] + scale * (posY - infile_begin[1]), scale * tile_size[0], scale * tile_size[1]]),
                    "infile": tuple([posX, posY, tile_size[0], tile_size[1]]),
                    "filename": self.plan["file"]
                }

                batchCounter += 1
                batch_images.append(image)
                batch_nodata.append(composite_nodata)
                batch_bounds.append(bounds)

                if batchCounter == self.batch_size:
                    return batch_images, batch_nodata, batch_bounds

            self.pos[1] += tile_step[1]
            self.pos[0] = infile_begin[0]

        if batchCounter > 0:
            return batch_images[:batchCounter], batch_nodata[:batchCounter], batch_bounds[:batchCounter]
        else:
            return None, None, None

    def __load_image(self):
        if self.skip:
            return

        ds = gdal.Open(self.plan["file"], gdal.GF_Read)

        begin_offset = self.plan["infile_begin"]
        end_offset = self.plan["infile_end"]
        end_offset = [end_offset[0] + 512, end_offset[1] + 512]

        size = [
            end_offset[0] - begin_offset[0],
            end_offset[1] - begin_offset[1]
        ]
        self.image_cache = np.zeros((size[1], size[0], len(self.bands) + 2), dtype="uint8")
        self.image_cache[:, :, -2:] = 1

        x_begin = max(-begin_offset[0], 0)
        x_end = min(size[0], ds.RasterXSize - begin_offset[0])

        y_begin = max(-begin_offset[1], 0)
        y_end = min(size[1], ds.RasterYSize - begin_offset[1])

        for i in range(len(self.bands)):
            ds_band = ds.GetRasterBand(self.bands[i])
            value = ds_band.ReadAsArray(max(0, begin_offset[0]), max(0, begin_offset[1]), x_end - x_begin, y_end - y_begin)

            self.image_cache[y_begin:y_end, x_begin:x_end, i] = np.clip(255 * ((value - self.min[i]) / (self.max[i] - self.min[i])), 0, 255).astype("uint8")
            if self.nodata_band is None:
                self.image_cache[y_begin:y_end, x_begin:x_end, -2] &= (value == self.nodata_value[i])

        if self.nodata_band is not None:
            ds_band = ds.GetRasterBand(self.nodata_band)
            value = ds_band.ReadAsArray(max(0, begin_offset[0]), max(0, begin_offset[1]), x_end - x_begin, y_end - y_begin)
            self.image_cache[y_begin:y_end, x_begin:x_end, -2] = (value == self.nodata_value)

        ds = None

        # 1 is valid data; 0 - nodata
        self.image_cache[:, :, -2] = 1 - self.image_cache[:, :, -2]
        self.image_cache[:, :, -1] = cv2.erode(self.image_cache[:, :, -2], np.ones((5, 5)))

    def __load_lclu(self, lclu_path):
        if self.skip or lclu_path is None:
            return

        lclu_ds = gdal.Open(lclu_path, gdal.GF_Read)

        scale = self.plan["scale"]
        begin_offset = [v // scale for v in self.plan["global_begin"]]
        end_offset = [(v + 512) // scale for v in self.plan["global_end"]]

        size = [
            end_offset[0] - begin_offset[0],
            end_offset[1] - begin_offset[1]
        ]

        x_begin = max(-begin_offset[0], 0)
        x_end = min(size[0], lclu_ds.RasterXSize - begin_offset[0])

        y_begin = max(-begin_offset[1], 0)
        y_end = min(size[1], lclu_ds.RasterYSize - begin_offset[1])

        lclu_band = lclu_ds.GetRasterBand(1)
        lclu = lclu_band.ReadAsArray(max(0, begin_offset[0]), max(0, begin_offset[1]), x_end - x_begin, y_end - y_begin)

        if self.lclu_range is None:
            for val in self.lclu_clip_values:
                self.image_cache[y_begin:y_end, x_begin:x_end, -2] &= (lclu != val)

            for val in self.lclu_filter_values:
                self.image_cache[y_begin:y_end, x_begin:x_end, -1] &= (lclu != val)
        else:
            clip_map = np.array([1 for _ in range(self.lclu_range)], dtype="uint8")
            filter_map = np.array([1 for _ in range(self.lclu_range)], dtype="uint8")

            for val in self.lclu_clip_values:
                clip_map[val] = 0

            for val in self.lclu_filter_values:
                filter_map[val] = 0

            self.image_cache[y_begin:y_end, x_begin:x_end, -2] &= clip_map[lclu]
            self.image_cache[y_begin:y_end, x_begin:x_end, -1] &= filter_map[lclu]