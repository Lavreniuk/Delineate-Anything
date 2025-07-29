import math
from osgeo import gdal

class ExecutionPlanner:
    def __init__(self, analyser, planner_config):
        self.analyser = analyser

        self.region_size_full = [
            # minx, miny, maxx, maxy = self.analyser.total_bounds
            self.analyser.scale * int(math.ceil((self.analyser.total_bounds[2] - self.analyser.total_bounds[0]) / self.analyser.pixel_size_x)),
            self.analyser.scale * int(math.ceil((self.analyser.total_bounds[3] - self.analyser.total_bounds[1]) / abs(self.analyser.pixel_size_y)))
        ]

        self.region_size = [
            planner_config["region_width"],
            planner_config["region_height"]
        ]

        self.pixel_offset = planner_config["pixel_offset"]

        self.current_region = None

    def get_geotransform(self):
        minx, _, _, maxy = self.analyser.total_bounds

        # Calculate pixel size and raster size
        pixel_width = self.analyser.pixel_size_x / self.analyser.scale
        pixel_height = self.analyser.pixel_size_y / self.analyser.scale

        # Calculate geotransform
        return (
            minx + self.current_region[0] * pixel_width,                             # top left x
            pixel_width,            # w-e pixel resolution
            0,                               # rotation (0 if north is up)
            maxy + self.current_region[1] * pixel_height,                             # top left y
            0,                               # rotation (0 if north is up)
            pixel_height           # n-s pixel resolution (negative value)
        )

    def get_num_regions(self):
        x = self.region_size_full[0] // self.region_size[0] + (1 if self.region_size_full[0] % self.region_size[0] != 0 else 0)
        y = self.region_size_full[1] // self.region_size[1] + (1 if self.region_size_full[1] % self.region_size[1] != 0 else 0)

        return x * y

    def move_to_next_region(self):
        if self.current_region is None:
            self.current_region = [0, 0]
        else:
            self.current_region[0] += self.region_size[0]
            if self.current_region[0] >= self.region_size_full[0]:
                self.current_region[0] = 0
                self.current_region[1] += self.region_size[1]
        
        return self.current_region[0] < self.region_size_full[0] and self.current_region[1] < self.region_size_full[1]

    def get_plan(self, tileSize, tileStep):
        if self.current_region is None:
            return []
        
        tileSize = self.analyser.tile_size if tileSize is None else tileSize
        
        tileStep = int(tileSize * tileStep)
        stepsPerNonIntersection = (tileSize // tileStep) + (1 if tileSize % tileStep != 0 else 0)
        
        plans = []
        for tiff in self.analyser.tiffs:
            ds = gdal.Open(tiff)
            ds_size = [ds.RasterXSize, ds.RasterYSize]
            ds_offset = self.analyser.get_pixel_offset(ds)

            left, right, top, bottom = [
                ds_offset[0], ds_offset[0] + ds_size[0],
                ds_offset[1], ds_offset[1] + ds_size[1]
            ]

            region_left, region_right, region_top, region_bottom = [
                self.current_region[0] // self.analyser.scale,
                (self.current_region[0] + self.region_size[0]) // self.analyser.scale,
                self.current_region[1] // self.analyser.scale,
                (self.current_region[1] + self.region_size[1]) // self.analyser.scale,
            ]

            # check if image extent intersects with region extent
            if not (left < region_right and right > region_left and top < region_bottom and bottom > region_top):
                continue

            begin_x = ((max(left, region_left) - (tileSize - tileStep)) // tileStep) * tileStep - left
            begin_y = ((max(top, region_top) - (tileSize - tileStep)) // tileStep) * tileStep - top

            end_x = min(right, region_right) - left
            end_y = min(bottom, region_bottom) - top

            # to ensure no overlaps within the region to not deal with race conditions
            for i in range(stepsPerNonIntersection):
                for j in range(stepsPerNonIntersection):
                    begin = [begin_x + i * tileStep + self.pixel_offset[0], begin_y + j * tileStep + self.pixel_offset[1]]
                    end = [end_x + self.pixel_offset[0], end_y + self.pixel_offset[1]]

                    if begin[0] >= end[0] or begin[1] >= end[1]:
                        continue

                    plans.append({
                        "file": tiff,
                        "infile_begin": begin,
                        "infile_end": end,
                        "infile_size": [tileSize, tileSize],
                        "infile_step": [stepsPerNonIntersection * tileStep, stepsPerNonIntersection * tileStep],
                        "inregion_begin": [self.analyser.scale * (begin_x + i * tileStep + left - region_left), self.analyser.scale * (begin_y + j * tileStep + top - region_top)],
                        "inregion_end": [self.analyser.scale * (end_x + left - region_left), self.analyser.scale * (end_y + top - region_top)],
                        "global_begin": [self.analyser.scale * (begin_x + i * tileStep + left), self.analyser.scale * (begin_y + j * tileStep + top)],
                        "global_end": [self.analyser.scale * (end_x + left), self.analyser.scale * (end_y + top)],
                        "scale": self.analyser.scale
                    })

        return plans