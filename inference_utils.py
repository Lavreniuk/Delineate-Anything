import os
import time
from osgeo import gdal
import numpy as np
from utils import *
from IDMapper import IDMapper
from TileLoader import TileLoader
import logging
import cv2
from tqdm import tqdm  # For progress bars

# Setup logger for better visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def delineate(model, tiff_path, field_id_counter, config):
    logger.info(f"Starting delineation of {tiff_path}")

    loader = TileLoader(
        tiff_path,
        tileDim=config["TILE_DIMENSIONS"],
        tileStep=config["TILE_STEP"],
        batchSize=config["BATCH_SIZE"],
        skip_tiles_with_nodata=config["SKIP_TILES_WITH_NODATA"],
        nodataValue=config["NODATA_VALUE"],
        alphaBand=config["ALPHA_BAND"]
    )

    src_ds = loader.dataSource
    scale_factor = 2 if loader.tileDim == 256 else 1
    boundary_width = (loader.tileDim - loader.tileStep) * scale_factor

    temp_instances_tiff_path = os.path.join(
        config["TEMP_FOLDER_PATH"],
        os.path.basename(tiff_path).replace(".tif", ".instances.tif")
    )
    temp_confidence_tiff_path = os.path.join(
        config["TEMP_FOLDER_PATH"],
        os.path.basename(tiff_path).replace(".tif", ".confidence.tif")
    )
    
    pixel_offset_x = config["PIXEL_OFFSET_X"]
    pixel_offset_y = config["PIXEL_OFFSET_Y"]
    instance_raster = create_tiff_with_same_bounds(
        temp_instances_tiff_path, src_ds, gdal.GDT_Int32, scale_factor, pixel_offset_x, pixel_offset_y
    )
    confidence_raster = create_tiff_with_same_bounds(
        temp_confidence_tiff_path, src_ds, gdal.GDT_Float32, scale_factor, pixel_offset_x, pixel_offset_y
    )

    instance_band = instance_raster.GetRasterBand(1)
    confidence_band = confidence_raster.GetRasterBand(1)

    total_time = 0
    dataloader_time = 0
    model_time = 0
    postproc_time = 0
    mapping_time = 0

    top_counter = 0
    mappings = {}

    # Try to get total_batches from the loader (if available)
    total_batches = None
    if hasattr(loader, 'estimated_batch_number'):
        total_batches = loader.estimated_batch_number

    # Use a tqdm progress bar for batch processing. If total_batches is None, the bar is indeterminate.
    with tqdm(total=total_batches, desc="Number of batches", unit=" batch") as pbar:
        while True:
            top_counter += 1
            t_start = time.time()
            batch_dict = loader.nextBatch()
            batch = batch_dict["images"]
            nodata = batch_dict["nodata"]
            bounds = batch_dict["bounds"]
            if batch is None:
                break
            
            t_end = time.time()
            dataloader_time += t_end - t_start
            total_time += t_end - t_start
            
            t_start = time.time()
            results = model.predict(batch, conf=config["MINIMAL_CONFIDENCE"], half=config["HALF"], verbose=False)
            t_end = time.time()
            model_time += t_end - t_start
            total_time += t_end - t_start

            for i, result in enumerate(results):
                if result.masks is None or len(result.masks) == 0:
                    continue
                
                t_start = time.time()
                nodata_mask = nodata[i].astype("uint8")
                nodata_mask = cv2.resize(nodata_mask, (512, 512), interpolation=cv2.INTER_NEAREST_EXACT)
                nodata_mask = cv2.dilate(nodata_mask, np.ones((5, 5)))
                nodata_mask = 1 - nodata_mask # invert to make 1 on data and 0 on nodata

                instance_map, confidence_map, field_id_counter = rasterize(
                    result, field_id_counter, 
                    config["PIXEL_AREA_THRESHOLD"], 
                    config["REMAINING_AREA_THRESHOLD"],
                    nodata_mask
                )

                lhs, width, ths, height = bounds[i]
                lhs *= scale_factor
                width *= scale_factor
                ths *= scale_factor
                height *= scale_factor

                existing_instances = instance_band.ReadAsArray(lhs + pixel_offset_x, ths + pixel_offset_y, width, height)
                existing_confidence = confidence_band.ReadAsArray(lhs + pixel_offset_x, ths + pixel_offset_y, width, height)
                new_instances = instance_map[:height, :width]
                new_confidence = confidence_map[:height, :width]

                mappings = find_edge_mapping(
                    existing_instances[:boundary_width, :],
                    new_instances[:boundary_width, :],
                    mappings,
                    config["MERGE_RELATIVE_AREA_THRESHOLD"],
                    config["MERGE_ASYMETRIC_MERGING_PIXEL_AREA_THRESHOLD"],
                    config["MERGE_ASYMETRYC_MERGING_RELATIVE_AREA_THRESHOLD"]
                )
                mappings = find_edge_mapping(
                    existing_instances[:, :boundary_width],
                    new_instances[:, :boundary_width],
                    mappings,
                    config["MERGE_RELATIVE_AREA_THRESHOLD"],
                    config["MERGE_ASYMETRIC_MERGING_PIXEL_AREA_THRESHOLD"],
                    config["MERGE_ASYMETRYC_MERGING_RELATIVE_AREA_THRESHOLD"]
                )

                write_instances = new_instances.copy()
                write_confidence = new_confidence.copy()
                mask = (new_confidence > existing_confidence).astype("bool")
                write_instances = new_instances * mask + existing_instances * (~mask)
                write_confidence = new_confidence * mask + existing_confidence * (~mask)

                instance_band.WriteArray(write_instances, lhs + pixel_offset_x, ths + pixel_offset_y)
                confidence_band.WriteArray(write_confidence, lhs + pixel_offset_x, ths + pixel_offset_y)
                t_end = time.time()
                postproc_time += t_end - t_start
                total_time += t_end - t_start

            # Update progress: if total_batches is known, the progress bar shows current batch / total.
            pbar.update(1)
    
    logger.info("Delineation has been finished.")

    # Merging: use a tqdm progress bar for merging rows
    t_start = time.time()
    logger.info("Preparing for merging...")
    mapper = IDMapper()
    for key in mappings.keys():
        arr = mappings[key]
        arr.append(key)
        mapper.union(arr)
    
    mapp = mapper.get_mapping()
    npmap = np.array([mapp[i] if i in mapp else i for i in range(field_id_counter + 1)])
    
    w = instance_raster.RasterXSize
    with tqdm(total=instance_raster.RasterYSize, desc="Merging rows", unit=" row") as pbar_merge:
        for i in range(instance_raster.RasterYSize):
            line = instance_band.ReadAsArray(0, i, w, 1)
            line = npmap[line]
            instance_band.WriteArray(line, 0, i)
            pbar_merge.update(1)
    
    t_end = time.time()
    mapping_time += t_end - t_start
    total_time += t_end - t_start

    logger.info(f"Fields have been merged. Total time: {round(total_time, 1)} s")
    logger.info(f"Data loader time: {round(dataloader_time, 1)} s")
    logger.info(f"Model time: {round(model_time, 1)} s")
    logger.info(f"Postprocess time: {round(postproc_time, 1)} s")
    logger.info(f"Merging (mapping) time: {round(mapping_time, 1)} s")
    logger.info("-------------------")
    
    confidence_raster.FlushCache()
    confidence_raster = None
    instance_raster.FlushCache()
    instance_raster = None
    
    return field_id_counter

def multiple_source_merge(paths, field_id_counter, config):
    total_time_start = time.time()

    extents = []
    for path in paths:
        file = os.path.basename(path)
        ds = gdal.Open(os.path.join(config["TEMP_FOLDER_PATH"], file.replace(".tif", ".instances.tif")))
        extents.append(get_extent(ds))
        projection = ds.GetProjection()
        pixel_size = ds.GetGeoTransform()[1]
    
    total_extent = merge_extents(extents)
    
    instances_output_path = config["OUTPUT_TEMP_RASTER_PATH"].replace(".tif", ".instances.tif")
    ds_instances = create_empty_tiff(instances_output_path, total_extent, projection, pixel_size, gdal.GDT_Int32)
    ds_confidence = create_empty_tiff(
        config["OUTPUT_TEMP_RASTER_PATH"].replace(".tif", ".confidence.tif"),
        total_extent, projection, pixel_size, gdal.GDT_Float32
    )
    
    band_instances = ds_instances.GetRasterBand(1)
    band_confidence = ds_confidence.GetRasterBand(1)
    
    mappings = {}
    # Use a tqdm progress bar for processing each file
    for path in tqdm(paths, desc="Processing files for merge", unit=" file"):
        local_time_start = time.time()
        file = os.path.basename(path)
        
        loader_instances = TileLoader(
            os.path.join(config["TEMP_FOLDER_PATH"], file.replace(".tif", ".instances.tif")),
            config["TILE_DIMENSIONS"], config["TILE_STEP"], 1, None, tileBands=1, dtype="int32"
        )
        loader_confidence = TileLoader(
            os.path.join(config["TEMP_FOLDER_PATH"], file.replace(".tif", ".confidence.tif")),
            config["TILE_DIMENSIONS"], config["TILE_STEP"], 1, None, tileBands=1, dtype="float32"
        )
        
        offset_x, offset_y = get_pixel_offset(ds_instances, loader_instances.dataSource)
        # Try to get total batches for this file, if available
        total_batches_file = None
        if hasattr(loader_instances, 'tiles'):
            total_batches_file = int(np.ceil(len(loader_instances.tiles) / config["BATCH_SIZE"]))
        
        with tqdm(total=total_batches_file, desc=f"Mapping {file}", unit=" batch") as pbar_file:
            while True:
                batch_instances = loader_instances.nextBatch()
                batch_confidence = loader_confidence.nextBatch()
                if batch_instances["images"] is None:
                    break
                
                instances = batch_instances["images"][0].squeeze(2)
                confidence = batch_confidence["images"][0].squeeze(2)
                lhs, width, ths, height = batch_confidence["bounds"][0]
                
                old_instances = band_instances.ReadAsArray(offset_x + lhs, offset_y + ths, width, height)
                old_confidence = band_confidence.ReadAsArray(offset_x + lhs, offset_y + ths, width, height)
                
                mappings = find_edge_mapping(
                    old_instances[:height, :width],
                    instances[:height, :width],
                    mappings,
                    config["MERGE_RELATIVE_AREA_THRESHOLD"],
                    config["MERGE_ASYMETRIC_MERGING_PIXEL_AREA_THRESHOLD"],
                    config["MERGE_ASYMETRYC_MERGING_RELATIVE_AREA_THRESHOLD"]
                )
                
                mask = (confidence[:height, :width] > old_confidence[:height, :width]).astype("bool")
                write_instances = instances[:height, :width] * mask + old_instances[:height, :width] * (~mask)
                write_confidence = confidence[:height, :width] * mask + old_confidence[:height, :width] * (~mask)
                
                band_instances.WriteArray(write_instances[:height, :width], offset_x + lhs, offset_y + ths)
                band_confidence.WriteArray(write_confidence[:height, :width], offset_x + lhs, offset_y + ths)
                pbar_file.update(1)
            # End of file batches loop
        
    logger.info("Images have been pre-merged.")
    
    mapper = IDMapper()
    for key in mappings.keys():
        arr = mappings[key]
        arr.append(key)
        mapper.union(arr)
    
    mapp = mapper.get_mapping()
    npmap = np.array([mapp[i] if i in mapp else i for i in range(field_id_counter + 1)])
    
    logger.info("Id mapping has been created.")
    
    w = ds_instances.RasterXSize
    with tqdm(total=ds_instances.RasterYSize, desc="Final merging rows", unit=" row") as pbar_merge:
        for i in range(ds_instances.RasterYSize):
            line = band_instances.ReadAsArray(0, i, w, 1)
            line = npmap[line]
            band_instances.WriteArray(line, 0, i)
            pbar_merge.update(1)
    
    ds_instances.FlushCache()
    ds_confidence.FlushCache()
    
    logger.info(f"Merging has been finished in {time.time() - total_time_start} s.")
    
    return instances_output_path

def progress_callback(complete, message, user_data):
    # Update tqdm progress bar if provided; otherwise, log info.
    if user_data is not None and isinstance(user_data, dict) and "pbar" in user_data:
        pbar = user_data["pbar"]
        pbar.n = int(complete * 100)
        pbar.refresh()
    else:
        logger.info(f"Polygonizing... {round(complete * 100, 1)} %")
    return 1

def polygonize(src_path, dst_path, config):
    t_start = time.time()
    
    instance_raster = gdal.Open(src_path)
    instance_band = instance_raster.GetRasterBand(1)
    
    # Setup a tqdm progress bar for polygonization (0 to 100%)
    with tqdm(total=100, desc="Polygonizing", unit="%") as pbar:
        user_data = {"pbar": pbar}
        gpkg, layer = create_geopackage_with_same_projection(
            dst_path, config["LAYER_NAME"], instance_raster,
            override_if_exists=config["OVERRIDE_IF_EXISTS"]
        )
        gdal.Polygonize(instance_band, instance_band, layer, -1, ["CONNECTED=4"],
                        callback=lambda complete, msg, u: progress_callback(complete, msg, user_data))
    
    instance_raster.FlushCache()
    instance_raster = None
    t_elapsed = time.time() - t_start
    logger.info(f"Polygonization has been completed in {t_elapsed} s.")
    cleanup_geopackage(layer, config["VECTORIZED_AREA_THRESHOLD"])
    logger.info("Geopackage cleaned up.")
