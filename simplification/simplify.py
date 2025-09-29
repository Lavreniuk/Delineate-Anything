from osgeo import ogr, osr
import multiprocessing

from .ReadWorker import ReadWorker
from .SimplificationWorker import SimplificationWorker
from .WriteWorker import WriteWorker

from multiprocessing.managers import BaseManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import math
from tqdm import tqdm

import time

import os, cv2
import rasterio
from rasterio.transform import from_bounds

ogr.UseExceptions()

# --- Step 1: Extend dict with batch fetch ---
class MyDict(dict):
    def get_many(self, keys):
        return {k: self[k] for k in keys if k in self}
    
    def get_many_as_array(self, keys):
        return [self[k] if k in self else 0 for k in keys]


# --- Step 2: Create a custom manager that serves MyDict ---
class MyManager(BaseManager):
    pass

# Register MyDict so proxies will have .get_many()
MyManager.register("MyDict", MyDict, 
                   exposed=[
        "__getitem__", "__setitem__", "__delitem__", "__contains__",
        "get", "keys", "values", "items", "update", "pop", "clear",
        "__iter__", "__len__", "get_many", "get_many_as_array"
    ])

def simplify(config):
    src_gpkg = config["src"]
    dst_gpkg = config["dst"]
    layer_name = config["layer_name"]
    
    VERTEX_STEP_X = "VERTEX_STEP_X"
    VERTEX_STEP_Y = "VERTEX_STEP_Y"

    ds = ogr.Open(src_gpkg, 1)
    metadata = ds.GetMetadata()
    vertex_step_x = abs(float(metadata[VERTEX_STEP_X]) / 2 if VERTEX_STEP_X in metadata else config["fallback_vertices_step"][0])
    vertex_step_y = abs(float(metadata[VERTEX_STEP_Y]) / 2 if VERTEX_STEP_Y in metadata else config["fallback_vertices_step"][1])

    superres_scale = config["superres_scale"]
    epsilon_scale = config["epsilon_scale"]
    epsilon = 2 * epsilon_scale * superres_scale * math.sqrt(vertex_step_x**2 + vertex_step_y**2)

    region_size = config["raster_resolution"]

    num_workers = config["num_workers"]
    if num_workers == -1:
        num_workers = max(1, multiprocessing.cpu_count() - 2)

    fid_column = config["unique_id_column"]
    if fid_column is None:
        GEN_FID = "GEN_FID"
        fid_column = GEN_FID

        layer = ds.GetLayerByName(layer_name)
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

        # Only create if it does not exist
        if GEN_FID not in field_names:
            field_defn = ogr.FieldDefn(GEN_FID, ogr.OFTInteger)
            layer.CreateField(field_defn)

            layer.StartTransaction()
            for feature in layer:
                fid = feature.GetFID()
                feature.SetField(GEN_FID, fid)
                layer.SetFeature(feature)

            layer.CommitTransaction()
            feature = None
    
    ds = None

    simplify_internal(src_gpkg, layer_name, dst_gpkg, layer_name, [vertex_step_x, vertex_step_y], epsilon, region_size, num_workers, fid_column)


def simplify_internal(src_gpkg, src_layer_name, dst_gpkg, dst_layer_name, densify_step, epsilon, region_size, num_workers, fid_column):
    clone_layer_schema(src_gpkg, src_layer_name, dst_gpkg, dst_layer_name, epsilon)
    
    dx = (region_size[0] - 1) * densify_step[0]
    dy = (region_size[1] - 1) * densify_step[1]
    width = (region_size[0] - 1) * densify_step[0]
    height = (region_size[1] - 1) * densify_step[1]

    ds = ogr.Open(src_gpkg, 0)
    layer = ds.GetLayerByName(src_layer_name)
    total_features = layer.GetFeatureCount()
    extent = layer.GetExtent()
    total_extent = [np.float64(extent[0]), np.float64(extent[1]), np.float64(extent[2]), np.float64(extent[3])]
    minx, _, miny, _ = total_extent
    maxx = minx + width
    maxy = miny + height

    num_worker_to_count = num_workers

    read_to_simplify_queue = multiprocessing.Queue(32768)
    simplify_to_write_queue = multiprocessing.Queue(32768)

    manager = MyManager()
    manager.start()

    incidence_matrix = SharedMemory(create=True, size=region_size[0]*region_size[1])
    incidence_np = np.ndarray((region_size[0], region_size[1]), dtype="uint8", buffer=incidence_matrix.buf)
    
    set_of_simplified_polygons = manager.MyDict()

    reader_feature_counter = multiprocessing.Value('i', 0)
    reader_start_flag = multiprocessing.Event()
    reader_done_flag = multiprocessing.Event()
    read_worker = ReadWorker(src_gpkg, src_layer_name, fid_column, read_to_simplify_queue, set_of_simplified_polygons, reader_start_flag, reader_done_flag, reader_feature_counter)
    read_worker.start()

    simplification_workers = [
        SimplificationWorker((incidence_matrix.name, region_size), densify_step, epsilon, read_to_simplify_queue, simplify_to_write_queue, set_of_simplified_polygons, multiprocessing.Value('i', 0)) for i in range(num_workers)
    ]

    for worker in simplification_workers:
        worker.start()

    writer_feature_counter = multiprocessing.Value('i', 0)
    write_worker = WriteWorker(dst_gpkg, dst_layer_name, simplify_to_write_queue, total_features, writer_feature_counter)
    write_worker.start()

    read_worker.started_event.wait()
    write_worker.started_event.wait()

    for worker in simplification_workers:
        worker.started_event.wait()

    counterr = 0
    while True:
        incidence_np[:, :] = 0
        block_extent = [minx, maxx, miny, maxy]

        reader_feature_counter.value = 0
        for worker in simplification_workers:
            worker.feature_counter.value = 0
        writer_feature_counter.value = 0
        
        for worker in simplification_workers[:num_worker_to_count]:
            worker.individual_input_queue.put((SimplificationWorker.MODE_COUNT_VERTICES, [minx, miny]))
            worker.individual_input_queue.join()

        reader_done_flag.clear()
        read_worker.input_queue.put((ReadWorker.MODE_READ_ALL, block_extent))
        reader_start_flag.set()
        reader_done_flag.wait()

        while True:
            count = 0
            for worker in simplification_workers:
                with worker.feature_counter.get_lock():
                    count += worker.feature_counter.value

            with reader_feature_counter.get_lock():
                if reader_feature_counter.value == count:
                    break

            time.sleep(0.1)

        reader_feature_counter.value = 0
        for worker in simplification_workers:
            worker.feature_counter.value = 0
        writer_feature_counter.value = 0

        # should be synchronous to update shared vertices counts
        for worker in simplification_workers[:num_worker_to_count]:
            with worker.feature_counter.get_lock():
                if worker.feature_counter == 0:
                    continue

            worker.individual_input_queue.put((SimplificationWorker.COMMAND_MERGE, None))
            worker.individual_input_queue.join()

        for worker in simplification_workers:
            worker.individual_input_queue.put((SimplificationWorker.MODE_SIMPLIFY, [minx, miny]))

        for worker in simplification_workers:
            worker.individual_input_queue.join()

        reader_done_flag.clear()
        # to make sure what polygons what touching original extent is pass WITHIN test
        read_worker.input_queue.put((ReadWorker.MODE_READ_NONSIMPLIFIED, block_extent))
        reader_start_flag.set()
        reader_done_flag.wait()

        while True:
            count = 0
            for worker in simplification_workers:
                with worker.feature_counter.get_lock():
                    count += worker.feature_counter.value

            with reader_feature_counter.get_lock(), writer_feature_counter.get_lock():
                # print(writer_feature_counter.value, "of", reader_feature_counter.value)
                if reader_feature_counter.value == writer_feature_counter.value:
                    break
                    
            time.sleep(0.1)

        for worker in simplification_workers:
            worker.individual_input_queue.put((SimplificationWorker.MODE_WAIT, None))

        for worker in simplification_workers:
            worker.individual_input_queue.join()

        minx += dx
        maxx = minx + width
        
        if minx >= total_extent[1]:
            miny += dy
            maxy = miny + height
            minx = total_extent[0]
            maxx = minx + width

            if miny >= total_extent[3]:
                break



    for worker in simplification_workers:
        worker.individual_input_queue.put((SimplificationWorker.MODE_TERMINATE, None))
    
    write_worker.input_queue.put(WriteWorker.MODE_TERMINATE)
    write_worker.join()

    for worker in simplification_workers:
        worker.join()

    incidence_matrix.close()
    incidence_matrix.unlink()

    read_worker.terminate()
    write_worker.terminate()
    for worker in simplification_workers:
        worker.terminate()

    dissolve_inplace(dst_gpkg, dst_layer_name, fid_column)

def clone_layer_schema(src_gpkg, src_layer_name, dst_gpkg, dst_layer_name, epsilon):
    driver = ogr.GetDriverByName("GPKG")

    src_ds = ogr.Open(src_gpkg, 0)
    if src_ds is None:
        raise RuntimeError(f"Cannot open source GeoPackage: {src_gpkg}")

    src_layer = src_ds.GetLayerByName(src_layer_name)
    if src_layer is None:
        raise RuntimeError(f"Layer {src_layer_name} not found in {src_gpkg}")

    src_defn = src_layer.GetLayerDefn()
    srs = src_layer.GetSpatialRef()

    dst_ds = driver.CreateDataSource(dst_gpkg)
    metadata = src_ds.GetMetadata()
    metadata["SIMPLIFICATION_EPSILON"] = epsilon 
    dst_ds.SetMetadata(metadata)

    if dst_layer_name is None:
        dst_layer_name = src_layer_name

    dst_layer = dst_ds.CreateLayer(dst_layer_name, srs, geom_type=ogr.wkbMultiPolygon)

    for i in range(src_defn.GetFieldCount()):
        field_defn = src_defn.GetFieldDefn(i)
        dst_layer.CreateField(field_defn)

    return dst_ds, dst_layer

def dissolve_inplace(input_gpkg, layername, field_name):
    drv = ogr.GetDriverByName("GPKG")
    ds = drv.Open(input_gpkg, 1)  # update mode
    if ds is None:
        raise RuntimeError(f"Could not open {input_gpkg}")
    layer = ds.GetLayerByName(layername)

    # --- Step 1: Count occurrences of each field value ---
    counts = {}
    for feat in tqdm(layer, total=layer.GetFeatureCount(), desc="Scanning"):
        val = feat.GetField(field_name)
        counts[val] = counts.get(val, 0) + 1

    # keep only those with >1
    multi_vals = {val for val, c in counts.items() if c > 1}
    if not multi_vals:
        ds = None
        return

    # --- Step 2: Collect geometries for those values ---
    groups = {val: None for val in multi_vals}
    fids_to_delete = []
    layer.ResetReading()

    for feat in tqdm(layer, total=layer.GetFeatureCount(), desc="Merging"):
        val = feat.GetField(field_name)
        if val in groups:
            geom = feat.GetGeometryRef()
            if geom:
                geom = geom.Clone()
                if groups[val] is None:
                    groups[val] = geom
                else:
                    groups[val] = groups[val].Union(geom)
            fids_to_delete.append(feat.GetFID())

    # --- Step 3 & 4 in one transaction ---
    layer.StartTransaction()

    # Delete old entries
    for fid in tqdm(fids_to_delete, desc="Deleting"):
        layer.DeleteFeature(fid)

    # Insert dissolved features
    defn = layer.GetLayerDefn()
    for val, geom in tqdm(groups.items(), desc="Inserting"):
        if geom is None:
            continue
        # force multipolygon
        if geom.GetGeometryType() != ogr.wkbMultiPolygon:
            geom = ogr.ForceToMultiPolygon(geom)

        new_feat = ogr.Feature(defn)
        new_feat.SetField(field_name, val)
        new_feat.SetGeometry(geom)
        layer.CreateFeature(new_feat)
        new_feat = None

    layer.CommitTransaction()

    layer.StartTransaction()

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(6933)

    source_srs = layer.GetSpatialRef()
    transform = osr.CoordinateTransformation(source_srs, target_srs)

    # Delete old entries
    for feat in tqdm(layer, total=layer.GetFeatureCount(), desc="Calc area"):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        # clone to avoid mutating the original
        geom_clone = geom.Clone()
        geom_clone.Transform(transform)

        # compute area in square meters
        area_val = geom_clone.GetArea()

        feat.SetField("area", area_val)
        layer.SetFeature(feat)

    layer.CommitTransaction()

    ds = None