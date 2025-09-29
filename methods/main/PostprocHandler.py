import numpy as np
import time

import multiprocessing
from multiprocessing import shared_memory

from .UnitedWorker import UnitedWorker
from .IDMapper import IncrementalFastMapper

from osgeo import ogr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        self.id_mapper = IncrementalFastMapper(10_000_000)

        # create queues
        self.queue = []# multiprocessing.Queue(self.queue_tiles_capacity)

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

        self.__create_workers()

    def put(self, args):
        while len(self.queue) == self.queue_tiles_capacity:
            self.run()

        self.queue.append(args)
        self.run()

    def sync(self):
        while not (len(self.queue) == 0 and self.tiles_inflight == 0):
            self.run()

    def run(self):
        # estimate load on each worker
        while not self.result_queue.empty():
            worker_id, local_mapping_dict = self.result_queue.get()
            self.tiles_inflight -= 1
            self.workers_load[worker_id] -= 1
            self.update_id_mapper(local_mapping_dict)

        if self.tiles_inflight >= self.max_tiles_inflight:
            return

        while not len(self.queue) == 0 and self.tiles_inflight < self.max_tiles_inflight:
            arg = self.queue.pop(0)
            argmin = np.argmin(self.workers_load)
            worker = self.workers_list[argmin]
            
            self.tiles_inflight += 1
            self.workers_load[argmin] += 1
            worker.queue.put((UnitedWorker.MODE_POSTPROC, arg, ((self.region_size[1], self.region_size[0]), (self.region_size[1], self.region_size[0]), self.postproc_config)))
    
    def set_postproc_config(self, config):
        self.postproc_config = config

    def clear(self):
        self.instances_map[:, :] = 0
        self.weights_map[:, :] = 0

    def update_id_mapper(self, local_mapping_dict):
        for key, val in local_mapping_dict.items():
            val.append(key)
            self.id_mapper.union(val)

    def map(self):
        start = time.time()
        npmap = np.array(self.id_mapper.finalize(), dtype="int32")
        end = time.time()

        self.instances_map[:, :] = npmap[self.instances_map]

        end_end = time.time()
        logger.debug(f"Mapped in {end_end - start} s; Applied in {end_end - end} s.")

    def apply_background(self, background):
        if background is None:
            return

        mask = self.instances_map < 2
        self.instances_map[mask] = -background[mask]

    def polygonize(self, geotransform, layer_info):
        t0 = time.time()
        gpkg_path, layer_name = layer_info

        workers_in_flight = 0
        di = self.instances_map.shape[0] // self.num_workers_grid[0]
        dj = self.instances_map.shape[1] // self.num_workers_grid[1]
        # setup and start polygonization workers
        for i in range(self.num_workers_grid[0]):
            for j in range(self.num_workers_grid[1]):
                local_gt = (
                    geotransform[0] + geotransform[1] * (j * dj),
                    geotransform[1],
                    0,
                    geotransform[3] + geotransform[5] * (i * di),
                    0,
                    geotransform[5]
                )

                worker = self.workers_grid[i][j]
                worker.queue.put((UnitedWorker.MODE_VECTORIZE, local_gt))
                workers_in_flight += 1


        gpkg = ogr.Open(gpkg_path, 1)
        out_layer = gpkg.GetLayerByName(layer_name)

        out_layer.StartTransaction()
        layer_defn = out_layer.GetLayerDefn()

        feature = ogr.Feature(layer_defn)
        try:
            while workers_in_flight > 0:
                if self.result_queue.empty():
                    time.sleep(0.01)

                while not self.result_queue.empty():
                    result = self.result_queue.get()
                    if result == None:
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

    def __create_workers(self):
        self.workers_list = [None] * self.num_workers
        self.workers_load = np.zeros((self.num_workers), dtype="int32")

        postproc_worker_args = (self.mapping_dict, self.area_dict)

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