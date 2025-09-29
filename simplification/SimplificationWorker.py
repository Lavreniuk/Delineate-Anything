import multiprocessing
import cv2
import numpy as np
from osgeo import ogr
from tqdm import tqdm

from multiprocessing.shared_memory import SharedMemory
from numba import njit, prange
from math import floor

import traceback

import time

class SimplificationWorker(multiprocessing.Process):
    MODE_TERMINATE = 0
    MODE_COUNT_VERTICES = 1
    MODE_SIMPLIFY = 2
    MODE_WAIT = 50
    COMMAND_MERGE = 100

    def __init__(self, incidence_matrix_info, step_size, epsilon, input_queue, output_queue, shared_set, feature_counter):
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

        self.feature_counter = feature_counter

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
                    try:
                        if mode == SimplificationWorker.MODE_COUNT_VERTICES:
                            self.count_vertices(args)
                            self.feature_counter.value += 1
                        elif mode == SimplificationWorker.MODE_SIMPLIFY:
                            self.simplify(args)
                            self.feature_counter.value += 1

                        continue
                    except:
                        traceback.print_exc()
                

            try:
                command, args = self.individual_input_queue.get_nowait()
                if command == SimplificationWorker.MODE_TERMINATE:
                    self.individual_input_queue.task_done()
                    break
                elif command == SimplificationWorker.COMMAND_MERGE:
                    self.merge_dicts()
                else:
                    mode = command
                    self.offset = args

                self.individual_input_queue.task_done()
            except:
                pass
            
            if mode == SimplificationWorker.MODE_WAIT:
                time.sleep(0.25)

        self.shm.close()

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
        _, wkb, _ = args

        geom = ogr.CreateGeometryFromWkb(wkb)
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
                out[i] = 1

    def simplify(self, args):
        fid, wkb, fields = args
        geom = ogr.CreateGeometryFromWkb(wkb)

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

            # i > 0 => hole => clockwise (CW), but we want all to be processed as CCW
            # if i > 0:
            #     keys.reverse()
            #     points.reverse()

            prev_is_edge = False
            vertices = []
            fixed = []

            for j in range(count):
                point = points[j % count]
                key = keys[j % count]

                # point otside of raster extent - skip it
                if key < 0:
                    print("invalid id")
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

                prev_incidence = incidences[(j - 1) % count]
                current_incidence = incidences[j % count]
                next_incidence = incidences[(j + 1) % count]

                if current_incidence > prev_incidence or current_incidence > next_incidence:
                    isAnchor = 1

                prev_is_edge = isEdge

                vertices.append((point[0], point[1]))
                if isAnchor > 0:
                    fixed.append(len(vertices) - isAnchor)

            if len(vertices) > 2:
                simplified = SimplificationWorker.simplify_with_fixed(vertices, self.epsilon, fixed)

                # if i > 0:
                #     simplified.reverse()

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
        try:
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

                vertices.append(p_start)
                keys.append(kstart)

                delta = (iend[0] - istart[0], iend[1] - istart[1])
                l = max(abs(delta[0]), abs(delta[1]))

                dense_step = (step[0] * np.sign(delta[0]), step[1] * np.sign(delta[1]))
                
                pos = p_start
                for _ in range(l - 1):
                    pos = (pos[0] + dense_step[0], pos[1] + dense_step[1])
                    key, _ = SimplificationWorker.to_key_and_ipos(pos, step, offset, dimx, dimy)

                    vertices.append(pos)
                    keys.append(key)
        except:
            traceback.print_exc()

        return vertices, keys
    
    @staticmethod
    def to_key_and_ipos(point, step, offset, dimx, dimy):
        # x = int(np.floor((point[0] - offset[0]) / step[0] + 0.5) + 0.5)
        # y = int(np.floor((point[1] - offset[1]) / step[1] + 0.5) + 0.5)

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

            # Avoid duplicate joins
            # if i > 0:
            #     approx = approx[1:]

            result.extend(approx.tolist())

        return result