import multiprocessing
import cv2
import numpy as np
from osgeo import ogr

from multiprocessing.shared_memory import SharedMemory
from numba import njit

from .ReadWorker import ReadWorker

import traceback
import time

class SimplificationWorker(multiprocessing.Process):
    MODE_TERMINATE = -1
    MODE_COUNT_VERTICES = -2
    MODE_SIMPLIFY = -3
    MODE_WAIT = -4

    def __init__(self, incidence_matrix_info, step_size, epsilon, output_queue):
        super().__init__(daemon=False)

        self.started_event = multiprocessing.Event()

        self.individual_input_queue = multiprocessing.JoinableQueue()
        self.output_queue = output_queue
        self.incidence_shm_name = incidence_matrix_info[0]
        self.incidence_dims = incidence_matrix_info[1] # Expected: (dimx, dimy)

        self.step_size = step_size
        self.epsilon = epsilon

        self.extent_geom = None

    def run(self):
        self.started_event.set()

        self.shm = SharedMemory(name=self.incidence_shm_name, create=False)
        # Allocate 1D array representing flat memory layout of (dimx * dimy * 4 directions)
        self.incidence_np = np.ndarray((self.incidence_dims[0] * self.incidence_dims[1] * 4), buffer=self.shm.buf, dtype="uint8")

        mode = SimplificationWorker.MODE_WAIT
        while True:
            try:
                args = self.individual_input_queue.get(timeout=0.1)
            except:
                continue

            if args[0] == SimplificationWorker.MODE_TERMINATE:
                self.individual_input_queue.task_done()
                break

            if args[0] == SimplificationWorker.MODE_WAIT:
                mode = SimplificationWorker.MODE_WAIT
                self.individual_input_queue.task_done()
            elif args[0] == SimplificationWorker.MODE_COUNT_VERTICES:
                mode = SimplificationWorker.MODE_COUNT_VERTICES
                self.offset, extent = args[1][0], args[1][1]
                self.extent_geom = ReadWorker.make_extent_geom(extent)
                self.individual_input_queue.task_done()
            elif args[0] == SimplificationWorker.MODE_SIMPLIFY:
                mode = SimplificationWorker.MODE_SIMPLIFY
                self.offset, extent = args[1][0], args[1][1]
                self.extent_geom = ReadWorker.make_extent_geom(extent)
                self.individual_input_queue.task_done()
            else:
                if mode == SimplificationWorker.MODE_WAIT:
                    time.sleep(0.1)
                    continue

                poly_args = []
                try:
                    poly_args = self.clip_geometry(args, self.extent_geom)
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

                self.individual_input_queue.task_done()

        self.shm.close()

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

    @staticmethod
    @njit(nogil=True)
    def apply_direct_updates(arr, keys, channels, l):
        """Writes simultaneously to the 1D flat array mapping spatial index and channel offset."""
        n = keys.shape[0]
        for i in range(n):
            k = keys[i]
            c = channels[i]
            if k >= 0:
                flat_k = k * 4 + c
                if 0 <= flat_k < l:
                    # theretically only once vertex can be approached from this direction -> we need to flag pixel noot increase it
                    arr[flat_k] = 1

    def count_vertices(self, args):
        _, geom, _ = args

        if geom is None:
            return

        # Explicitly force OGC standard orientation: Outer = CCW, Holes = CW
        oriented_geom = ogr.ForceToPolygon(geom)
        # if oriented_geom is None:
        #     oriented_geom = geom # Fallback if forcing fails

        for i in range(oriented_geom.GetGeometryCount()):
            ring = oriented_geom.GetGeometryRef(i)
            
            if ring.GetPointCount() < 3:
                continue

            _, keys, channels = SimplificationWorker.densify(
                ring, self.step_size, self.offset, self.incidence_dims[0], self.incidence_dims[1]
            )

            if len(keys) == 0:
                continue

            keys_np = np.array(keys, dtype="int64")
            channels_np = np.array(channels, dtype="uint8")

            try:
                SimplificationWorker.apply_direct_updates(
                    self.incidence_np, keys_np, channels_np, self.incidence_np.shape[0]
                )
            except:
                traceback.print_exc()

    @staticmethod
    @njit(nogil=True)
    def gather_incidence(arr, keys, out, l):
        """Sums up all 4 directional channels for each spatial key index to return total count."""
        n = keys.shape[0]
        for i in range(n):
            k = keys[i]
            if k >= 0 and (k * 4 + 3) < l:
                base_idx = k * 4
                out[i] = (
                    arr[base_idx] + 
                    arr[base_idx + 1] + 
                    arr[base_idx + 2] + 
                    arr[base_idx + 3]
                )
            else:
                out[i] = 0

    def simplify(self, args):
        fid, geom, fields = args

        if geom is None:
            self.output_queue.put((fid, None, None))
            return

        empty = True
        new_polygon = ogr.Geometry(ogr.wkbPolygon)

        for i in range(geom.GetGeometryCount()):
            ring = geom.GetGeometryRef(i)
            # Uses original, raw topologies safely since gather_incidence combines the channel vectors
            points, keys, _ = SimplificationWorker.densify(ring, self.step_size, self.offset, self.incidence_dims[0], self.incidence_dims[1])
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

                if key < 0:
                    prev_is_edge = False
                    continue
                
                isAnchor = 0
                isEdge = False
                if key < self.incidence_dims[0] or (key + 1) % self.incidence_dims[0] == 0 or key % self.incidence_dims[0] == 0 or key >= (self.incidence_dims[1] - 1) * self.incidence_dims[0]:
                    isEdge = True 
                
                if not prev_is_edge and isEdge:
                    isAnchor = 1
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
                simplified = SimplificationWorker.simplify_with_fixed(simplified, 1e-3 * self.epsilon, [])

                if len(simplified) > 2:
                    # convert from pixel-space to crs space
                    simplified = [(self.offset[0] + self.step_size[0] * p[0], self.offset[1] + self.step_size[1] * p[1]) for p in simplified]

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
        """Traces the ring path and categorizes every vertex into 1 of 4 directional tracks."""
        vertices = []
        keys = []
        channels = []

        initial_vertices_count = ring.GetPointCount()
        # was initial_vertices_count - 1, but in case we dont have duplication of last vertex we will use initial_vertices_count, 
        # if duplication is present it will be just skipped by "if l == 0:"
        for i in range(initial_vertices_count):
            i_start = i
            i_end = (i + 1) % initial_vertices_count

            p_start = np.float64(ring.GetPoint(i_start))
            p_end = np.float64(ring.GetPoint(i_end))

            kstart, istart = SimplificationWorker.to_key_and_ipos((p_start[0], p_start[1]), step, offset, dimx, dimy)
            _, iend = SimplificationWorker.to_key_and_ipos((p_end[0], p_end[1]), step, offset, dimx, dimy)

            delta = (iend[0] - istart[0], iend[1] - istart[1])
            l = max(abs(delta[0]), abs(delta[1]))

            # it should not happen, but lets be safe
            if l == 0:
                continue

            # Determine dominant direction vector of the current line segment
            # 0: East (+X), 1: South (+Y), 2: West (-X), 3: North (-Y)
            if abs(delta[0]) >= abs(delta[1]):
                channel = 0 if delta[0] >= 0 else 2
            else:
                channel = 1 if delta[1] >= 0 else 3

            vertices.append(istart)
            keys.append(kstart)
            channels.append(channel)
            
            dense_step = (np.sign(delta[0]), np.sign(delta[1]))
            pos = istart
            for _ in range(l - 1):
                 # we no longer need to call to_key_and_ipos, as we are working with pixel space coordinates
                pos = (pos[0] + dense_step[0], pos[1] + dense_step[1])

                vertices.append(pos)
                keys.append(np.int64(dimx * pos[1] + pos[0]))
                channels.append(channel)

        return vertices, keys, channels
    
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

        n = len(fixed_indices) if closed else len(fixed_indices) - 1

        for i in range(n):
            start = fixed_indices[i]
            end = fixed_indices[(i + 1) % len(fixed_indices)]

            if start == end:
                continue

            if start < end:
                segment = points[start:end + 1]
            else:
                segment = points[start:] + points[:end + 1]

            p_start = segment[0]
            p_end = segment[-1]

            need_to_reverse = (p_end[1] < p_start[1]) or (p_end[1] == p_start[1] and p_end[0] < p_start[0])
            if need_to_reverse:
                segment = segment[::-1]

            arr = np.array(segment, dtype=np.float32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(arr, epsilon, False).reshape(-1, 2)

            approx[0] = segment[0]
            approx[-1] = segment[-1]

            if need_to_reverse:
                approx = approx[::-1]

            result.extend(approx.tolist())

        return result