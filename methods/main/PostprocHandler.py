import numpy as np
import time
import queue

import multiprocessing
from multiprocessing import shared_memory

from .UnitedWorker import UnitedWorker
from .IDMapper import IncrementalFastMapper

from osgeo import ogr
import logging

import cv2
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# half width (px) of the image strips kept along tile borders to compare the colour on both sides of a seam
SEAM_STRIP = 8

class PostprocHandler:
    def __init__(self, region_size, limits, srs, polygonization_config):
        # init params
        self.region_size = region_size

        self.num_workers_grid = limits["num_workers"]
        self.num_workers = self.num_workers_grid[0] * self.num_workers_grid[1]

        self.queue_tiles_capacity = limits["queue_tiles_capacity"]
        self.max_tiles_inflight = limits["max_tiles_inflight"]

        self.srs_wkt = srs
        self.config_poly = polygonization_config
        
        self.tiles_inflight = 0

        # region-space positions of tile borders (x of left/right, y of top/bottom edges)
        self.tile_edges_x = set()
        self.tile_edges_y = set()
        # ("x", pos) -> uint8 (H, 2 * SEAM_STRIP, C) image strip around a vertical border; ("y", pos) -> (2 * SEAM_STRIP, W, C)
        self.seam_strips = {}

        self.id_mapper = IncrementalFastMapper(10_000_000)

        # create queues
        self.queue = []

        # create shared dictionaries
        self.manager = multiprocessing.Manager()
        self.mapping_dict = self.manager.dict({})
        self.area_dict = self.manager.dict({})

        # create shared raster targets
        instances_raster_byte_size = region_size[0] * region_size[1] * 4
        self.instances_shared_memory = shared_memory.SharedMemory(create=True, size=instances_raster_byte_size)
        self.instances_map = np.ndarray(shape=(region_size[1], region_size[0]), dtype="int32", buffer=self.instances_shared_memory.buf)

        weights_raster_byte_size = region_size[0] * region_size[1] * 4
        self.weights_shared_memory = shared_memory.SharedMemory(create=True, size=weights_raster_byte_size)
        self.weights_map = np.ndarray(shape=(region_size[1], region_size[0]), dtype="float32", buffer=self.weights_shared_memory.buf)

        # fields as seen by the tile centred on each pixel's area, i.e. by the tile centred on the tile borders running
        # through it (see PostprocWorker.write_centre_view)
        self.seam_shared_memory = shared_memory.SharedMemory(create=True, size=region_size[0] * region_size[1] * 4)
        self.centre_map = np.ndarray(shape=(region_size[1], region_size[0]), dtype="int32", buffer=self.seam_shared_memory.buf)
        self.centre_map[:, :] = 0

        self.__create_workers()

    def put(self, args, image=None):
        bx, by, bw, bh = (int(v) for v in args[2]["inregion"])
        self.tile_edges_x.update((bx, bx + bw))
        self.tile_edges_y.update((by, by + bh))

        if image is not None and self.postproc_config.get("merge_tile_seams", True):
            self.__store_seam_strips(image, bx, by, bw, bh)

        while len(self.queue) == self.queue_tiles_capacity:
            if not self.run():
                time.sleep(0.001)

        self.queue.append(args)
        self.run()

    def sync(self):
        while not (len(self.queue) == 0 and self.tiles_inflight == 0):
            if not self.run():
                time.sleep(0.001)

    def run(self):
        made_progress = False
        # estimate load on each worker
        while True:
            try:
                worker_id, local_mapping_dict = self.result_queue.get_nowait()
            except queue.Empty:
                break

            self.tiles_inflight -= 1
            self.workers_load[worker_id] -= 1
            self.update_id_mapper(local_mapping_dict)
            made_progress = True

        if self.tiles_inflight >= self.max_tiles_inflight:
            return made_progress

        while not len(self.queue) == 0 and self.tiles_inflight < self.max_tiles_inflight:
            arg = self.queue.pop(0)
            argmin = np.argmin(self.workers_load)
            worker = self.workers_list[argmin]
            
            self.tiles_inflight += 1
            self.workers_load[argmin] += 1
            worker.queue.put((UnitedWorker.MODE_POSTPROC, arg, ((self.region_size[1], self.region_size[0]), (self.region_size[1], self.region_size[0]), self.postproc_config)))
            made_progress = True

        return made_progress
    
    def set_postproc_config(self, config):
        self.postproc_config = config

    def clear(self):
        self.instances_map[:, :] = 0
        self.weights_map[:, :] = 0
        # ids are globally unique and never reused by later regions
        self.area_dict.clear()
        self.tile_edges_x.clear()
        self.tile_edges_y.clear()
        self.seam_strips.clear()
        self.centre_map[:, :] = 0

    def update_id_mapper(self, local_mapping_dict):
        for key, val in local_mapping_dict.items():
            val.append(key)
            self.id_mapper.union(val)

    def map(self):
        start = time.time()
        npmap = np.array(self.id_mapper.finalize(), dtype="int32")
        end = time.time()

        self.instances_map[:, :] = npmap[self.instances_map]

        if self.postproc_config.get("merge_tile_seams", True):
            self.__merge_tile_seams()

        PostprocHandler.__id_opening(self.instances_map)

        end_end = time.time()
        logger.debug(f"Mapped in {end_end - start} s; Applied in {end_end - end} s.")

    def __merge_tile_seams(self):
        # A field that neighbouring tiles failed to merge ends up as two ids meeting exactly on a tile border.
        # Two ids are joined only if, on that border, they touch along most of their cut edges, the tile centred on
        # the border did not see two different fields there, and the image looks the same on both sides.
        min_contact = self.postproc_config.get("seam_min_contact_px", 8)
        min_coverage = self.postproc_config.get("seam_min_coverage", 0.3)
        min_evidence = self.postproc_config.get("seam_min_evidence", 0.5)
        max_colour_diff = self.postproc_config.get("seam_max_colour_diff", 0.2)
        # the tile centred on the border is asked whether it sees a boundary within this many px of the border
        self.split_window = self.postproc_config.get("seam_split_window_px", 1)
        # the two cut edges must start and end at the same place (within this many px, or this share of the contact),
        # unless the smaller piece is only a thin strip along the border (at most seam_max_strip_depth_px deep)
        self.end_tolerance = (self.postproc_config.get("seam_end_tolerance_px", 8), self.postproc_config.get("seam_end_tolerance_rel", 0.25))
        self.max_strip_depth = self.postproc_config.get("seam_max_strip_depth_px", 16)

        pairs = self.__find_seam_pairs(min_contact, min_coverage, min_evidence, max_colour_diff)
        if len(pairs) == 0:
            return

        for a, b in pairs:
            self.id_mapper.union([a, b])
        npmap = np.array(self.id_mapper.finalize(), dtype="int32")
        self.instances_map[:, :] = npmap[self.instances_map]
        logger.debug(f"Merged {len(pairs)} field pairs split by tile borders.")

    def __store_seam_strips(self, image, bx, by, bw, bh):
        # keep the image around every tile border crossing this tile (its own borders and, with a half-tile step,
        # its centre lines); overlapping tiles fill the other side of each border
        if image.shape[0] != bh or image.shape[1] != bw:
            return
        height, width = self.instances_map.shape
        channels = image.shape[2]

        r0, r1 = max(by, 0), min(by + bh, height)
        for x in (bx, bx + bw // 2, bx + bw):
            c0, c1 = max(x - SEAM_STRIP, bx, 0), min(x + SEAM_STRIP, bx + bw, width)
            if not (0 < x < width) or c0 >= c1 or r0 >= r1:
                continue
            strip = self.seam_strips.get(("x", x))
            if strip is None:
                strip = self.seam_strips[("x", x)] = np.zeros((height, 2 * SEAM_STRIP, channels), dtype="uint8")
            strip[r0:r1, c0 - (x - SEAM_STRIP):c1 - (x - SEAM_STRIP)] = image[r0 - by:r1 - by, c0 - bx:c1 - bx]

        c0, c1 = max(bx, 0), min(bx + bw, width)
        for y in (by, by + bh // 2, by + bh):
            r0, r1 = max(y - SEAM_STRIP, by, 0), min(y + SEAM_STRIP, by + bh, height)
            if not (0 < y < height) or c0 >= c1 or r0 >= r1:
                continue
            strip = self.seam_strips.get(("y", y))
            if strip is None:
                strip = self.seam_strips[("y", y)] = np.zeros((2 * SEAM_STRIP, width, channels), dtype="uint8")
            strip[r0 - (y - SEAM_STRIP):r1 - (y - SEAM_STRIP), c0:c1] = image[r0 - by:r1 - by, c0 - bx:c1 - bx]

    def __find_seam_pairs(self, min_contact, min_coverage, min_evidence, max_colour_diff):
        data, centre = self.instances_map, self.centre_map
        height, width = data.shape
        k = SEAM_STRIP // 2   # depth (px) on each side of the border used for the colour comparison
        pairs = []
        for x in sorted(self.tile_edges_x):
            if not (0 < x < width):
                continue
            line_pairs = []
            strip = self.seam_strips.get(("x", x))
            crossing, split = PostprocHandler.__centre_evidence(centre, x, self.split_window)
            for id_a, id_b, rows, aligned in PostprocHandler.__seam_pairs_on_line(data[:, x - 1], data[:, x], crossing, split,
                                                                                  min_contact, min_coverage, min_evidence, self.end_tolerance):
                if not aligned:
                    depth = self.max_strip_depth + 1
                    depth_a = PostprocHandler.__depth(data[rows, max(0, x - depth):x][:, ::-1], id_a)
                    depth_b = PostprocHandler.__depth(data[rows, x:x + depth], id_b)
                    if min(depth_a, depth_b) > self.max_strip_depth:
                        continue
                if strip is not None and x - k >= 0 and x + k <= width:
                    ids_a, ids_b = data[rows, x - k:x], data[rows, x:x + k]
                    img_a, img_b = strip[rows, SEAM_STRIP - k:SEAM_STRIP], strip[rows, SEAM_STRIP:SEAM_STRIP + k]
                    if PostprocHandler.__colour_diff(img_a[ids_a == id_a], img_b[ids_b == id_b]) > max_colour_diff:
                        continue
                line_pairs.append((id_a, id_b, len(rows)))
            pairs += PostprocHandler.__mutual_best(line_pairs)
        for y in sorted(self.tile_edges_y):
            if not (0 < y < height):
                continue
            line_pairs = []
            strip = self.seam_strips.get(("y", y))
            crossing, split = PostprocHandler.__centre_evidence(centre.T, y, self.split_window)
            for id_a, id_b, cols, aligned in PostprocHandler.__seam_pairs_on_line(data[y - 1, :], data[y, :], crossing, split,
                                                                                  min_contact, min_coverage, min_evidence, self.end_tolerance):
                if not aligned:
                    depth = self.max_strip_depth + 1
                    depth_a = PostprocHandler.__depth(data[max(0, y - depth):y, cols][::-1, :].T, id_a)
                    depth_b = PostprocHandler.__depth(data[y:y + depth, cols].T, id_b)
                    if min(depth_a, depth_b) > self.max_strip_depth:
                        continue
                if strip is not None and y - k >= 0 and y + k <= height:
                    ids_a, ids_b = data[y - k:y, cols], data[y:y + k, cols]
                    img_a, img_b = strip[SEAM_STRIP - k:SEAM_STRIP, cols], strip[SEAM_STRIP:SEAM_STRIP + k, cols]
                    if PostprocHandler.__colour_diff(img_a[ids_a == id_a], img_b[ids_b == id_b]) > max_colour_diff:
                        continue
                line_pairs.append((id_a, id_b, len(cols)))
            pairs += PostprocHandler.__mutual_best(line_pairs)
        return pairs

    @staticmethod
    def __centre_evidence(centre, x, window):
        # What the tile centred on the border at column x saw around it, per position along the border: a boundary
        # (two different fields, or a gap between fields) within window px of the border, or one field across the
        # whole window. A real boundary is seen there by that tile too, even when it is a few px off the border;
        # a field cut only by the tiling is not (whatever other splits the model makes elsewhere in the field).
        block = centre[:, max(x - window, 0):x + window]
        field = block >= 2
        big = np.iinfo(block.dtype).max
        lowest = np.where(field, block, big).min(axis=1)
        highest = np.where(field, block, -1).max(axis=1)
        two_fields = field.any(axis=1) & (lowest != highest)
        gap = field[:, 0] & field[:, -1] & ~field.all(axis=1)
        return field.all(axis=1) & ~two_fields, two_fields | gap

    @staticmethod
    def __mutual_best(line_pairs):
        # a field cut by a border has exactly one counterpart on the other side; keep a pair only if each piece is the
        # other's best partner (longest contact) on this border, so a thin strip can't join two fields together
        best_a, best_b = {}, {}
        for id_a, id_b, contact in line_pairs:
            if contact > best_a.get(id_a, (None, -1))[1]:
                best_a[id_a] = (id_b, contact)
            if contact > best_b.get(id_b, (None, -1))[1]:
                best_b[id_b] = (id_a, contact)
        return [(id_a, id_b) for id_a, id_b, _ in line_pairs if best_a[id_a][0] == id_b and best_b[id_b][0] == id_a]

    @staticmethod
    def __depth(block, field_id):
        # block: one row per contact position, columns going away from the border; how far the field reaches (max over rows)
        inside = block == field_id
        if inside.size == 0:
            return 0
        runs = np.where(inside.all(axis=1), inside.shape[1], np.argmin(inside, axis=1))
        return int(runs.max())

    @staticmethod
    def __colour_diff(pixels_a, pixels_b):
        # distance between the mean colours (0-1 scale) of the two sides of a seam; a field cut by a tile border
        # looks the same on both sides, two different fields usually don't
        if len(pixels_a) == 0 or len(pixels_b) == 0:
            return 0.0
        return float(np.linalg.norm(pixels_a.mean(axis=0) - pixels_b.mean(axis=0)) / 255.0)

    @staticmethod
    def __run_bounds(mask, first, last):
        # extent of the continuous run of True in mask that contains positions first..last
        start, end = first, last
        while start > 0 and mask[start - 1]:
            start -= 1
        while end + 1 < len(mask) and mask[end + 1]:
            end += 1
        return start, end

    @staticmethod
    def __seam_pairs_on_line(side_a, side_b, crossing, split, min_contact, min_coverage, min_evidence, end_tolerance):
        # side_a / side_b: ids in the pixel rows (or columns) right before and right after a tile border;
        # crossing / split: what the tile centred on this border saw there (one field crosses / two fields meet)
        touching = (side_a >= 2) & (side_b >= 2) & (side_a != side_b)
        if not touching.any():
            return []

        keys, inverse, contacts = np.unique((side_a[touching].astype(np.int64) << 32) | side_b[touching].astype(np.int64),
                                            return_inverse=True, return_counts=True)
        crossing_count = np.bincount(inverse, weights=(crossing[touching] > 0), minlength=len(keys))
        split_count = np.bincount(inverse, weights=(split[touching] > 0), minlength=len(keys))

        ids_a, len_a = np.unique(side_a[side_a >= 2], return_counts=True)
        ids_b, len_b = np.unique(side_b[side_b >= 2], return_counts=True)
        len_a, len_b = dict(zip(ids_a, len_a)), dict(zip(ids_b, len_b))

        touching_index = np.nonzero(touching)[0]
        pairs = []
        for i, (key, contact, n_crossing, n_split) in enumerate(zip(keys, contacts, crossing_count, split_count)):
            id_a, id_b = int(key >> 32), int(key & 0xFFFFFFFF)
            # both cut edges must be mostly in contact: a real split field has matching cuts on both sides
            if contact < min_contact or contact < min_coverage * max(len_a[id_a], len_b[id_b]):
                continue
            # the tile that saw this border from its centre must not have seen two different fields there
            # (where it saw no field at all, only the geometry decides)
            if n_split > 0 and n_crossing < min_evidence * (n_crossing + n_split):
                continue
            # positions along the border where the two ids touch
            positions = touching_index[inverse == i]
            # a field cut by the border continues across it: both cut edges start and end at the same place,
            # while two different fields meeting on the border usually have edges of different extent
            first, last = int(positions.min()), int(positions.max())
            a0, a1 = PostprocHandler.__run_bounds(side_a == id_a, first, last)
            b0, b1 = PostprocHandler.__run_bounds(side_b == id_b, first, last)
            tolerance = max(end_tolerance[0], end_tolerance[1] * contact)
            aligned = abs(a0 - b0) <= tolerance and abs(a1 - b1) <= tolerance
            pairs.append((id_a, id_b, positions, aligned))
        return pairs

    @staticmethod
    def __id_opening(data):
        chunk_size = 2048
        halo = 2
        
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        for i in range(0, data.shape[0], chunk_size):
            for j in range(0, data.shape[1], chunk_size):
                i_min, i_max = max(0, i - halo), min(data.shape[0], i + chunk_size + halo)
                j_min, j_max = max(0, j - halo), min(data.shape[1], j + chunk_size + halo)
                
                frag = data[i_min:i_max, j_min:j_max].copy()

                # 1. nullify border region
                mx = ndimage.maximum_filter(frag, footprint=kernel)
                mn = ndimage.minimum_filter(frag, footprint=kernel)
                frag[mx != mn] = 0

                # 2. fill gap
                mx = ndimage.maximum_filter(frag, footprint=kernel)
                frag[frag == 0] = mx[frag == 0]

                # 3. fill gap 2
                mx = ndimage.maximum_filter(frag, footprint=kernel)
                frag[frag == 0] = mx[frag == 0]

                # 4. remove overgrowth
                zero_mask = (frag == 0).astype(np.uint8)
                expanded_zeros = cv2.dilate(zero_mask, kernel.astype(np.uint8), borderType=cv2.BORDER_REPLICATE)
                frag[expanded_zeros > 0] = 0

                # 2. Determine slice to write back (skip the halo)
                write_i_start = halo if i > 0 else 0
                write_j_start = halo if j > 0 else 0
                
                actual_w = min(chunk_size, data.shape[0] - i)
                actual_h = min(chunk_size, data.shape[1] - j)

                data[i:i+actual_w, j:j+actual_h] = frag[write_i_start:write_i_start+actual_w, 
                                                        write_j_start:write_j_start+actual_h]

    def apply_background(self, background):
        if background is None:
            return

        mask = self.instances_map < 2
        self.instances_map[mask] = -background[mask]

    def polygonize(self, base_geotransform, region_offset, layer_info):
        t0 = time.time()
        gpkg_path, layer_name = layer_info

        workers_in_flight = 0
        # setup and start polygonization workers; each worker shifts its pixel geometry by an exact
        # integer offset and applies the same global geotransform, so shared edges get identical coordinates
        for i in range(self.num_workers_grid[0]):
            for j in range(self.num_workers_grid[1]):
                worker = self.workers_grid[i][j]
                worker.queue.put((UnitedWorker.MODE_VECTORIZE, (base_geotransform, tuple(region_offset))))
                workers_in_flight += 1


        gpkg = ogr.Open(gpkg_path, 1)
        out_layer = gpkg.GetLayerByName(layer_name)

        out_layer.StartTransaction()
        layer_defn = out_layer.GetLayerDefn()

        feature = ogr.Feature(layer_defn)
        try:
            while workers_in_flight > 0:
                result = self.result_queue.get()
                if result is None:
                    workers_in_flight -= 1
                    continue

                wkb, area, geom_id, isBackground = result
                geom = ogr.CreateGeometryFromWkb(wkb)

                feature.SetFID(-1)
                feature.SetGeometry(geom)
                feature.SetField("id", geom_id)
                feature.SetField("bg", isBackground)
                feature.SetField("area", float(area))

                out_layer.CreateFeature(feature)

            out_layer.CommitTransaction()
        except Exception as e:
            out_layer.RollbackTransaction()
            raise e

        logger.debug(f"Polygonization finished in {time.time() - t0} s.")

    def dispose(self):
        for worker in self.workers_list:
            worker.queue.put(UnitedWorker.MODE_TERMINATE)

        for worker in self.workers_list:
            worker.join()

        self.instances_shared_memory.close()
        self.instances_shared_memory.unlink()

        self.weights_shared_memory.close()
        self.weights_shared_memory.unlink()

        self.seam_shared_memory.close()
        self.seam_shared_memory.unlink()

    def __create_workers(self):
        self.workers_list = [None] * self.num_workers
        self.workers_load = np.zeros((self.num_workers), dtype="int32")

        postproc_worker_args = (self.mapping_dict, self.area_dict, self.seam_shared_memory.name)

        self.result_queue = multiprocessing.Queue()
        self.workers_grid = [[None]*self.num_workers_grid[1] for _ in range(self.num_workers_grid[0])]
        di = self.instances_map.shape[0] // self.num_workers_grid[0]
        dj = self.instances_map.shape[1] // self.num_workers_grid[1]
        for i in range(self.num_workers_grid[0]):
            begin_i = di * i
            end_i = (begin_i + di) if (i + 1) < self.num_workers_grid[0] else self.instances_map.shape[0] 
            for j in range(self.num_workers_grid[1]):
                begin_j = dj * j
                end_j = (begin_j + dj) if (j + 1) < self.num_workers_grid[1] else self.instances_map.shape[1] 

                process_id = i * self.num_workers_grid[1] + j
                vectorize_worker_args = (self.instances_map.shape, (begin_i, begin_j), (end_i, end_j), self.config_poly, self.srs_wkt)
                process = UnitedWorker(process_id, self.result_queue, self.instances_shared_memory.name, self.weights_shared_memory.name, 
                                       postproc_worker_args, vectorize_worker_args)
                process.start()
                self.workers_grid[i][j] = process
                self.workers_list[process_id] = process
                self.workers_load[process_id] = 0
                
        for i in range(self.num_workers):
            self.workers_list[i].start_running.wait()