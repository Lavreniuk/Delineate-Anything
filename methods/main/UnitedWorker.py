import multiprocessing
from multiprocessing import shared_memory
import time

from .PostprocWorker import PostprocWorker
from .PolygonizationWorker import PolygonizationWorker

class UnitedWorker(multiprocessing.Process):
    MODE_TERMINATE = 0
    MODE_POSTPROC = 1
    MODE_VECTORIZE = 2

    def __init__(self, id, result_queue, shm_instances_name, shm_weight_name, postproc_args, polygonize_args):
        super().__init__(daemon=False)

        self.queue = multiprocessing.Queue()
        self.start_running = multiprocessing.Event()

        self.id = id

        self.result_queue = result_queue
        self.shm_instances = shared_memory.SharedMemory(name=shm_instances_name)
        self.shm_weights = shared_memory.SharedMemory(name=shm_weight_name)

        # set postproc related args
        mapping_dict, area_dict = postproc_args
        self.mapping_dict = mapping_dict
        self.area_dict = area_dict

        # set vectorization related args
        raster_shape, region_begin, region_end, poly_config, srs_wkt = polygonize_args
        self.raster_shape = raster_shape
        self.region_begin = region_begin
        self.region_end = region_end
        self.poly_config = poly_config
        self.srs_wkt = srs_wkt

    def run(self):
        self.start_running.set()

        inner_queue = []

        instances_shape = None
        weight_shape = None
        postproc_config = None
        while True:
            while not self.queue.empty():
                arg = self.queue.get()

                if arg == UnitedWorker.MODE_TERMINATE:
                    self.shm_instances.close()
                    self.shm_weights.close()
                    return
                
                mode = arg[0]
                if mode == UnitedWorker.MODE_POSTPROC:
                    _, args_1, args_2 = arg
                    model_results, nodata, bounds, id_counter = args_1

                    entries = PostprocWorker.first_wave_postprocessing(model_results, nodata, bounds, id_counter, 2)
                    inner_queue.append(entries)

                    instances_shape, weight_shape, postproc_config = args_2

                if mode == UnitedWorker.MODE_VECTORIZE:
                    _, vectorize_arg = arg
                    PolygonizationWorker.polygonize(self.result_queue, self.shm_instances, self.raster_shape, vectorize_arg, self.poly_config, self.srs_wkt, self.region_begin, self.region_end)
                    self.result_queue.put(None)

            # mode == UnitedWorker.MODE_POSTPROC
            if len(inner_queue) > 0:
                entries = inner_queue.pop(0)
                local_mapping_dict = {}
                PostprocWorker.process_fields(entries, local_mapping_dict, self.area_dict, (self.shm_instances, instances_shape), (self.shm_weights, weight_shape), postproc_config)
                self.result_queue.put((self.id, local_mapping_dict))