from ultralytics import YOLO
import os
import time
import logging
from inference_utils import *
from utils import *
from osgeo import gdal, osr, ogr
import shutil
import math

import torch
import torchvision

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# Configuration Constants - Define all the configuration variables here for easy access
MODEL_WEIGHT_PATH = "test.pt"
SRC_FOLDER = "images"
AUTO_CROP_RESOLUTION_SELECTION = True
TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "result"
OUTPUT_TEMP_RASTER_PATH = os.path.join(OUTPUT_FOLDER, "result.tif")
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "result.gpkg")

# Configuration for delineation
delineation_config = {
    "TILE_DIMENSIONS": 256,  # 256 or 512, otherwise unexpected behavior might occur
    "TILE_STEP": 192,        # Step size between tiles
    "SKIP_TILES_WITH_NODATA": False,  # If True, skips tiles containing any NoData value
    "ALPHA_BAND": None,      # GDAL number of the band to use as alpha, or None if not applicable
    "NODATA_VALUE": 0,       # The value considered as NoData
    "BATCH_SIZE": 4,         # Batch size for processing

    "TEMP_FOLDER_PATH": TEMP_FOLDER,   # Temporary folder for processing files
    "DELETE_TEMP_FILES": True,
    "PIXEL_OFFSET_X": 2,        # Horizontal pixel offset
    "PIXEL_OFFSET_Y": 2,        # Vertical pixel offset
    "MINIMAL_CONFIDENCE": 0.175,  # Minimum confidence for object detection
    "HALF": False,

    "PIXEL_AREA_THRESHOLD": 64,  # Minimum pixel area for delineated objects
    "REMAINING_AREA_THRESHOLD": 0.3,  # Threshold for remaining areas after processing

    "MERGE_RELATIVE_AREA_THRESHOLD": 0.4,  # Relative overlap area between fields A and B for merging
    "MERGE_ASYMETRIC_MERGING_PIXEL_AREA_THRESHOLD": 128,  # Minimum pixel area for asymmetric merging
    "MERGE_ASYMETRYC_MERGING_RELATIVE_AREA_THRESHOLD": 0.85,  # Relative overlap area for asymmetric merging
}

# Configuration for interimage merging
interimage_merging_config = {
    "TILE_DIMENSIONS": 128,  # Tile dimensions for merging
    "TILE_STEP": 128,        # Step size for merging tiles
    "MERGE_RELATIVE_AREA_THRESHOLD": 0.4,  # Relative area threshold for merging
    "MERGE_ASYMETRIC_MERGING_PIXEL_AREA_THRESHOLD": 128,  # Minimum pixel area for asymmetric merging
    "MERGE_ASYMETRYC_MERGING_RELATIVE_AREA_THRESHOLD": 0.85,  # Relative area for asymmetric merging
    "TEMP_FOLDER_PATH": TEMP_FOLDER,
    "OUTPUT_TEMP_RASTER_PATH": OUTPUT_TEMP_RASTER_PATH
}

# Configuration for polygonization
polygonize_config = {
    "LAYER_NAME": "fields",        # Layer name in the output vector file
    "OVERRIDE_IF_EXISTS": True,    # If True, overwrites the layer if it already exists
    "VECTORIZED_AREA_THRESHOLD": 5000  # Minimum area (in projected units) to vectorize
}


def create_output_folder(folder_name):
    """Ensure the output folder exists."""
    if not os.path.exists(folder_name):
        logger.info(f"Creating output folder: {folder_name}")
        os.makedirs(folder_name)


def check_compatibility(src_folder):
    """Check the compatibility of TIFF files (projections, pixel sizes)."""
    tiff_paths = []
    projections = []
    pixel_sizes = []

    for file in os.listdir(src_folder):
        if not file.endswith(".tif"):
            continue
        
        path = os.path.join(src_folder, file)
        tiff_paths.append(path)
        ds = gdal.Open(path)
        projections.append(ds.GetProjection())
        pixel_sizes.append(ds.GetGeoTransform()[1])
        ds = None

    if not tiff_paths:
        logger.error("No TIFF files found in the source folder.")
        return None, None, None

    # Check if all projections and pixel sizes are the same
    assert all(p == projections[0] for p in projections), "ERROR! Projections are not the same."
    assert all(p == pixel_sizes[0] for p in pixel_sizes), "ERROR! Pixels sizes are not the same."

    return tiff_paths, projections, pixel_sizes

def get_pixel_size_meters(tiff_path):
    # Open the dataset
    ds = gdal.Open(tiff_path)
    gt = ds.GetGeoTransform()
    width_units = abs(gt[1])
    height_units = abs(gt[5])
    
    # Get center coordinates
    width = ds.RasterXSize
    height = ds.RasterYSize
    center_y = gt[3] + (width / 2) * gt[4] + (height / 2) * gt[5]

    # Load projection
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
 
    if srs.IsProjected():
        scale = srs.GetLinearUnits()  # meters per unit

        width_meters = width_units * scale
        height_meters = height_units * scale
    else:
        scale = srs.GetAngularUnits() # radians per unit
        lat_rad = center_y * scale # latitude of center in radians
        earth_radius = 6371000  # meters
        lat_m = earth_radius * scale 
        lon_m = lat_m * math.cos(lat_rad)

        width_meters = width_units * lon_m
        height_meters = height_units * lat_m

    return 0.5 * (width_meters + height_meters)

def main():
    time_start = time.time()  # Initialize time_start for execution time tracking

    # Ensure the output folder exists
    create_output_folder(TEMP_FOLDER)
    create_output_folder(OUTPUT_FOLDER)

    logger.info("Checking compatibility...")
    tiff_paths, projections, pixel_sizes = check_compatibility(SRC_FOLDER)

    if not tiff_paths:
        logger.error("No TIFF files found. Early stopping.")
        return

    logger.info("Loading the model...")
    model = YOLO(MODEL_WEIGHT_PATH).to(device)
    logger.info("Model has been loaded.")

    time_delineate_start = time.time()
    global_field_counter = 0
    for path in tiff_paths:
        if AUTO_CROP_RESOLUTION_SELECTION:
            approx_meters = get_pixel_size_meters(path)
            print("Approximate pixel size:", approx_meters, "m")
            if approx_meters > 5:
                delineation_config["TILE_DIMENSIONS"] = 256
                delineation_config["TILE_STEP"] = 192
            else:
                delineation_config["TILE_DIMENSIONS"] = 512
                delineation_config["TILE_STEP"] = 384

        global_field_counter = delineate(model, path, global_field_counter, delineation_config)

    logger.info(f"All images have been delineated in {time.time() - time_delineate_start:.2f} seconds.")

    # Merging the images if there are multiple TIFF files
    if len(tiff_paths) > 1:
        raster_instances_path = multiple_source_merge(tiff_paths, global_field_counter, interimage_merging_config)
    else:
        logger.info("Merging has been skipped. Only one image to process.")
        raster_instances_path = os.path.join(TEMP_FOLDER, os.path.basename(tiff_paths[0]).replace(".tif", ".instances.tif"))

    # Polygonizing the results
    polygonize(raster_instances_path, OUTPUT_PATH, polygonize_config)
    if delineation_config["DELETE_TEMP_FILES"]:
        shutil.rmtree(delineation_config["TEMP_FOLDER_PATH"])
        if os.path.exists(raster_instances_path):
            os.remove(raster_instances_path)
        raster_confidence_path = raster_instances_path.replace(".instances.tif", ".confidence.tif")
        if os.path.exists(raster_confidence_path):
            os.remove(raster_confidence_path)
        logger.info("Temporary files and folder have been deleted.")
    
    logger.info(f"Execution finished in {time.time() - time_start:.2f} seconds.")


if __name__ == "__main__":
    main()
