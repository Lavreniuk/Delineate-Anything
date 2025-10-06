import multiprocessing
import cv2
import numpy as np
from osgeo import ogr

from multiprocessing.shared_memory import SharedMemory
from numba import njit, prange

from .ReadWorker import ReadWorker

import traceback

import time

class SimplificationWorker(multiprocessing.Process):
    MODE_TERMINATE = 0
    MODE_COUNT_VERTICES = 1
    MODE_SIMPLIFY = 2
    MODE_WAIT = 50
    COMMAND_MERGE = 100

    def __init__(self, incidence_matrix_info, step_size, epsilon, input_queue, output_queue, shared_set, reader_feature_counter, writer_feature_counter):
        super().__init__(daemon=False)

        self.started_event = multiprocessing.Event()

        self.input_queue = input_queue
        self.individual_input_queue = multiprocessing.JoinableQueue()
        self.output_queue = output_queue
        self.local_vertices_dict = {}
        self.shared_set = shared_set
        self.incidence_shm_name = incidence_matrix_info[0]
        self.incidence_dims = incidence_matrix_info[1]

        self.step_size = step_size
        self.epsilon = epsilon

        self.reader_feature_counter = reader_feature_counter
        self.writer_feature_counter = writer_feature_counter

        self.extent_geom = None

    def run(self):
        self.started_event.set()

        self.shm = SharedMemory(name=self.incidence_shm_name, create=False)
        self.incidence_np = np.ndarray((self.incidence_dims[0] * self.incidence_dims[1]), buffer=self.shm.buf, dtype="uint8")

        mode = SimplificationWorker.MODE_WAIT
        while True:
            if mode != SimplificationWorker.MODE_WAIT:
                hasContent = False
                try:
                    args = self.input_queue.get(timeout=0.1)
                    hasContent = True
                except:
                    hasContent = False

                if hasContent:
                    poly_args = []
                    try:
                        poly_args = self.clip_geometry(args, self.extent_geom)
                        self.reader_feature_counter.value += 1
                    except:
                            traceback.print_exc()

                    for poly in poly_args:
                        try:
                            if mode == SimplificationWorker.MODE_COUNT_VERTICES:
                                self.count_vertices(poly)
                                
                            elif mode == SimplificationWorker.MODE_SIMPLIFY:
                                self.simplify(poly)
                        except:
                            traceback.print_exc()

                    continue
  
            try:
                command, args = self.individual_input_queue.get_nowait()
                mode = self.process_individual_command(command, args, mode)
                self.individual_input_queue.task_done()
            except:
                pass
            
            if mode == SimplificationWorker.MODE_TERMINATE:
                break

            if mode == SimplificationWorker.MODE_WAIT:
                time.sleep(0.25)

        self.shm.close()

    def process_individual_command(self, command, args, mode):
        if command is None:
            return mode

        if command == SimplificationWorker.MODE_TERMINATE:
            self.individual_input_queue.task_done()
        elif command == SimplificationWorker.COMMAND_MERGE:
            self.merge_dicts()
            return mode
        else:
            if args is None:
                return mode
            
            self.offset, extent = args[0], args[1]
            self.extent_geom = ReadWorker.make_extent_geom(extent)

        return command

    def clip_geometry(self, args, extent_geom):
            output = []

            fid, wkb, fields = args
            geom = ogr.CreateGeometryFromWkb(wkb)

            clipped_geom = extent_geom.Intersection(geom).Buffer(0)
            if clipped_geom is None or clipped_geom.IsEmpty():
                return [(fid, None, fields)]

            if not clipped_geom.IsValid():
                return [(fid, None, fields)]

            def handle_any_geom(geom):
                if geom.GetGeometryName() == "POLYGON":
                    output.append((fid, geom, fields))
                    return

                count = geom.GetGeometryCount()
                for i in range(count):
                    sub = geom.GetGeometryRef(i)
                    handle_any_geom(sub)

            handle_any_geom(clipped_geom)
            return output

    def merge_dicts(self):
        keys = np.array(list(self.local_vertices_dict.keys()), dtype="int64")
        values = np.array(list(self.local_vertices_dict.values()), dtype="uint8")

        try:
            SimplificationWorker.apply_updates(self.incidence_np, keys, values, self.incidence_np.shape[0])
        except:
            traceback.print_exc()

        self.local_vertices_dict.clear()

    @staticmethod
    @njit(parallel=True)
    def apply_updates(arr, keys, vals, l):
        n = keys.shape[0]
        for i in prange(n):
            k = keys[i]
            if k >= 0 and k < l:
                arr[k] += vals[i]

    def count_vertices(self, args):
        _, geom, _ = args

        if geom is None:
            return

        for i in range(geom.GetGeometryCount()):
            ring = geom.GetGeometryRef(i)
            _, keys = SimplificationWorker.densify(ring, self.step_size, self.offset, self.incidence_dims[0], self.incidence_dims[1])

            for j in range(len(keys)):
                key = keys[j]
                if key < 0:
                    continue

                if key in self.local_vertices_dict:
                    self.local_vertices_dict[key] += 1
                else:
                    self.local_vertices_dict[key] = 1

    @staticmethod
    @njit(parallel=True)
    def gather_incidence(arr, keys, out, l):
        n = keys.shape[0]
        for i in prange(n):
            k = keys[i]
            if k >= 0 and k < l:
                out[i] = arr[k]
            else:
                out[i] = 0

    def simplify(self, args):
        fid, geom, fields = args
        self.writer_feature_counter.value += 1

        if geom is None:
            self.output_queue.put((fid, None, None))
            return

        empty = True
        new_polygon = ogr.Geometry(ogr.wkbPolygon)

        for i in range(geom.GetGeometryCount()):
            ring = geom.GetGeometryRef(i)
            points, keys = SimplificationWorker.densify(ring, self.step_size, self.offset, self.incidence_dims[0], self.incidence_dims[1])
            count = len(points)

            if len(points) < 3:
                continue

            try:
                incidences = np.empty((len(keys)), dtype="uint8")
                SimplificationWorker.gather_incidence(self.incidence_np, np.array(keys, dtype="int64"), incidences, self.incidence_np.shape[0])
            except:
                traceback.print_exc()

            prev_is_edge = False
            vertices = []
            fixed = []
            start_j = 0
            start_pos = points[0]
            for j in range(1, count):
                p = points[j]
                if p[0] > start_pos[0] or (p[0] == start_pos[0] and p[1] > start_pos[1]):
                    start_j = j
                    start_pos = p

            for j in range(start_j, start_j + count):
                point = points[j % count]
                key = keys[j % count]

                # point otside of raster extent - skip it
                if key < 0:
                    prev_is_edge = False
                    continue
                
                isAnchor = 0

                isEdge = False
                # on top, right, left, or bottom edge
                if key < self.incidence_dims[0] or (key + 1) % self.incidence_dims[0] == 0 or key % self.incidence_dims[0] == 0 or key >= (self.incidence_dims[1] - 1) * self.incidence_dims[0]:
                    isEdge = True 
                
                # first time touching edge
                if not prev_is_edge and isEdge:
                    isAnchor = 1
                    
                # we exited edge, so we must put anchor on previous position
                elif prev_is_edge and not isEdge:
                    isAnchor = 2

                vertices.append((point[0], point[1]))
                if isAnchor > 0:
                    fixed.append(len(vertices) - isAnchor)
                    continue

                prev_incidence = incidences[(j - 1) % count]
                current_incidence = incidences[j % count]
                next_incidence = incidences[(j + 1) % count]

                if current_incidence > prev_incidence or current_incidence > next_incidence:
                    isAnchor = 1

                prev_is_edge = isEdge

                if isAnchor > 0:
                    fixed.append(len(vertices) - isAnchor)

            if len(vertices) > 2:
                simplified = SimplificationWorker.simplify_with_fixed(vertices, self.epsilon, fixed)

                if len(simplified) > 2:
                    new_ring = ogr.Geometry(ogr.wkbLinearRing)
                    for x, y in simplified:
                        new_ring.AddPoint_2D(x, y)
                    new_ring.CloseRings()

                    new_polygon.AddGeometry(new_ring)
                    empty = False

        if not empty:
            fixed = new_polygon.Buffer(0)
            if fixed.GetGeometryType() == ogr.wkbPolygon:
                multipoly = ogr.Geometry(ogr.wkbMultiPolygon)
                multipoly.AddGeometry(fixed)
                fixed = multipoly

            output_wkb = fixed.ExportToWkb()
            self.output_queue.put((fid, output_wkb, fields))
        else:
            self.output_queue.put((-1, None, None))


    @staticmethod
    def densify(ring, step, offset, dimx, dimy):
        vertices = []
        keys = []

        initial_vertices_count = ring.GetPointCount()
        for i in range(initial_vertices_count - 1):
            i_start = i
            i_end = (i + 1) % initial_vertices_count

            p_start = np.float64(ring.GetPoint(i_start))
            p_start = (p_start[0], p_start[1])
            p_end = np.float64(ring.GetPoint(i_end))
            p_end = (p_end[0], p_end[1])

            kstart, istart = SimplificationWorker.to_key_and_ipos(p_start, step, offset, dimx, dimy)
            _, iend = SimplificationWorker.to_key_and_ipos(p_end, step, offset, dimx, dimy)

            delta = (iend[0] - istart[0], iend[1] - istart[1])
            l = max(abs(delta[0]), abs(delta[1]))

            if l > 1:
                vertices.append(p_start)
                keys.append(kstart)

                dense_step = (step[0] * np.sign(delta[0]), step[1] * np.sign(delta[1]))
                pos = p_start
                for _ in range(l - 1):
                    pos = (pos[0] + dense_step[0], pos[1] + dense_step[1])
                    key, _ = SimplificationWorker.to_key_and_ipos(pos, step, offset, dimx, dimy)

                    vertices.append(pos)
                    keys.append(key)

        return vertices, keys
    
    @staticmethod
    def to_key_and_ipos(point, step, offset, dimx, dimy):
        x = int(np.round((point[0] - offset[0]) / step[0]) + 0.5)
        y = int(np.round((point[1] - offset[1]) / step[1]) + 0.5)

        if (x < 0 or x >= dimx) or (y < 0 or y >= dimy):
            return np.int64(-1), (x, y)

        return np.int64(dimx * y + x), (x, y)

    @staticmethod
    def simplify_with_fixed(points, epsilon, fixed_indices, closed=True):
        if len(fixed_indices) < 2:
            arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(arr, epsilon, True).reshape(-1, 2)
            return approx.tolist()

        fixed_indices = sorted(set(fixed_indices))
        result = []

        # number of segments to process:
        n = len(fixed_indices) if closed else len(fixed_indices) - 1

        for i in range(n):
            start = fixed_indices[i]
            end = fixed_indices[(i + 1) % len(fixed_indices)]  # wrap for last→first

            if start == end:
                continue

            if start < end:
                segment = points[start:end + 1]
            else:  # wrap-around case
                segment = points[start:] + points[:end + 1]

            p_start = segment[0]
            p_end = segment[-1]

            # top-left comparison: smaller y first, then smaller x
            need_to_reverse = (p_end[1] < p_start[1]) or (p_end[1] == p_start[1] and p_end[0] < p_start[0])
            if need_to_reverse:
                segment = segment[::-1]

            # Convert to OpenCV-compatible array
            arr = np.array(segment, dtype=np.float32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(arr, epsilon, False).reshape(-1, 2)

            # Force fixed endpoints
            approx[0] = segment[0]
            approx[-1] = segment[-1]

            if need_to_reverse:
                approx = approx[::-1]

            result.extend(approx.tolist())

        return result