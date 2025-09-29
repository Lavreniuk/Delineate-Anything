from ultralytics import YOLO
import os
import time
import logging
from tqdm import tqdm

from .utils import *
from .DataAnalyser import DataAnalyser
from .ExecutionPlanner import ExecutionPlanner
from .DataLoaderCached import DataLoaderCached
from . import PostprocHandler as PostprocHandlerLib
from .PostprocHandler import PostprocHandler
from .PolygonizationWorker import PolygonizationWorker
from .BackgroundLoader import BackgroundLoader

from simplification import simplify

from osgeo import gdal, osr, ogr
import shutil
import math
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_output_folder(folder_name):
    """Ensure the output folder exists."""
    if not os.path.exists(folder_name):
        logger.info(f"Creating output folder: {folder_name}")
        os.makedirs(folder_name)

def execute(model_paths, config, verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)
        PostprocHandlerLib.logger.setLevel(logging.DEBUG)

    src_folder, temp_folder, output_path, keep_temp, mask_filepath = (
        config["execution_args"]["src_folder"],
        config["execution_args"]["temp_folder"],
        config["execution_args"]["output_path"],
        config["execution_args"]["keep_temp"],
        config["execution_args"]["mask_filepath"]
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device:", device)

    time_start = time.time()  # Initialize time_start for execution time tracking

    # Ensure the output folder exists
    create_output_folder(temp_folder)
    create_output_folder(os.path.dirname(output_path))

    tiffs = [os.path.join(src_folder, file) for file in os.listdir(src_folder) if (file.endswith(".tif") or file.endswith(".tiff"))]
    if config["treat_as_vrt"]:
        vrt_path = os.path.join(temp_folder, os.path.basename(src_folder) + ".vrt")
        gdal.BuildVRT(vrt_path, tiffs)
        tiffs = [vrt_path]

    print(tiffs)

    analyser = DataAnalyser(tiffs, config["data_loader"]["bands"], config["super_resolution"])
    if not analyser.isCompatible():
        logger.error(f"Incompatible tiff files. Ensure the same projection and pixel size fo each file in the folder.")
        return

    logger.info("Estimating normalization bounds...")
    analyser.calcNormalizationBounds()

    config["data_loader"]["min"] = analyser.min
    config["data_loader"]["max"] = analyser.max

    planner = ExecutionPlanner(analyser, config["execution_planner"])

    logger.info("Loading the model...")
    models = []
    for model_path in model_paths:
        model = YOLO(model_path).to(device)
        models.append(model)

    model_names = config["model"]
    models = {model_names[i]: models[i] for i in range(len(model_names))}

    lclu_mask_path = warp_lclu(mask_filepath, os.path.join(temp_folder, os.path.basename(src_folder) + ".lclu.tif"), 
                               tiffs[0], analyser.total_bounds, [analyser.pixel_size_x, analyser.pixel_size_y], 
                               ["BIGTIFF=YES", "COMPRESS=ZSTD", "ZSTD_LEVEL=2", "TILED=YES", "NUM_THREADS=ALL_CPUS"])

    gpkg_path, layer_name = create_geopackage_with_same_projection(
            output_path, config["polygonization_args"]["layer_name"], analyser.projection,
            override_if_exists=config["polygonization_args"]["override_if_exists"],
            pixel_size=[analyser.pixel_size_x / analyser.scale, analyser.pixel_size_y / analyser.scale]
        )

    time_delineate_start = time.time()
    logger.info("Starting delineation...")
    execute_delineation(models, planner, config["postprocess_limits"], config["passes"], config["data_loader"], 
                        (gpkg_path, layer_name), lclu_mask_path, config["mask_info"], config)
    
    logger.info(f"All regions have been delineated in {time.time() - time_delineate_start:.2f} seconds.")

    postdelineation_merge((gpkg_path, layer_name), config["filtering_args"])

    if not keep_temp:
        folder_content = os.listdir(temp_folder)
        for file in folder_content:
            if file.startswith(os.path.basename(src_folder) + "."):
                os.remove(os.path.join(temp_folder, file))

        if len(os.listdir(temp_folder)) == 0:
            shutil.rmtree(temp_folder)

        logger.info("Temporary files have been deleted.")
    
    logger.info(f"Delineation finished in {time.time() - time_start:.2f} seconds.")

    if config["simplification_args"]["simplify"] == True:
        start = time.time()
        logger.info("Simplification started")
        execute_simplification(gpkg_path, layer_name, config["simplification_args"], analyser.scale)
        logger.info(f"Simplification finished in {time.time() - start:.2f} seconds")

def execute_simplification(gpkg_path, layer_name, config, scale):
    full_config = {
        "src": gpkg_path,
        "dst": gpkg_path.replace(".gpkg", ".simp.gpkg"),
        "superres_scale": scale,
        "epsilon_scale": config["epsilon_scale"],
        # how many processes spawn for the task. -1 - all available cpus
        "num_workers": config["num_workers"],
        # 1 byte per pixel => 65k*65k = 4GB. for delineated with default parameters S2 data it effectively corresponds to square with 164km side.
        "raster_resolution": config["raster_resolution"],

        # no reason to touch it if you use delineation result using our code
        "layer_name": layer_name,
        # can't be fid; null = generate; if specified column is not unique - simplification will stuck
        "unique_id_column": "id"
    }

    simplify.simplify(full_config)



def execute_delineation(models, planner, postproc_config, passes, dataloader_config, layer_info, lclu_path, lclu_config, full_config):
    gpkg = ogr.Open(layer_info[0], 1)
    out_layer = gpkg.GetLayerByName(layer_info[1])
    srs = out_layer.GetSpatialRef()
    srs_wkt = srs.ExportToWkt()

    background_loader = BackgroundLoader(full_config["background_info"], lclu_path, lclu_config["range"])
    full_config["filtering_args"]["middleground_offset"] = background_loader.offset
    postproc_handler = PostprocHandler(planner.region_size, postproc_config, srs_wkt, full_config["filtering_args"])
    

    global_field_counter = 2
    field_counter_increment = 2

    counter_history_array = [{} for _ in passes]
    region_counter = 0

    total_dataloader_time = 0
    dataloader = None
    num_regions = planner.get_num_regions()
    with tqdm(total=num_regions, desc="Delineating", unit="region") as pbar_delineate:
        pbar_delineate.n = region_counter
        pbar_delineate.refresh()

        while planner.move_to_next_region():
            start_start = time.time()
            for pass_id in range(len(passes)):
                pass_args = passes[pass_id].copy()
                pass_args["delineation_config"]["region_offset"] = planner.current_region

                postproc_handler.set_postproc_config(pass_args["delineation_config"])

                tileSize = pass_args["tile_size"]
                tileStep = pass_args["tile_step"]
                batchSize = pass_args["batch_size"]

                plan = planner.get_plan(tileSize, tileStep)
                for entry in plan:
                    start = time.time()

                    is_compatible = dataloader is not None and dataloader.is_compatible(entry)
                    if dataloader is None or not is_compatible:
                        dataloader = DataLoaderCached(entry, dataloader_config, batchSize, lclu_path, lclu_config)
                        dt = time.time() - start
                        logger.debug(f"Data loader load data in: {dt} s.")
                        total_dataloader_time += dt
                    else:
                        dataloader.set_plan(entry)
                    
                    while True:
                        images_batch, nodata_batch, bounds_batch = dataloader.get_batch()

                        if images_batch is None:
                            break

                        model_results = []
                        for model_args in pass_args["model_args"]:
                            model = models[model_args["name"]]
                            prediction = model.predict(images_batch, conf=model_args["minimal_confidence"], half=model_args["use_half"], verbose=False)
                            model_results.append(prediction)

                        # push each batch item into queue
                        for i in range(len(images_batch)):
                            lbound = bounds_batch[i]
                            tile_key = tuple([lbound["filename"], lbound["infile"]])

                            counter_history = counter_history_array[pass_id]
                            id_counter = global_field_counter
                            if tile_key in counter_history:
                                id_counter = counter_history[tile_key]
                            else:
                                counter_history[tile_key] = id_counter
                                for result in model_results:
                                    if result[i].masks is not None:
                                        global_field_counter += field_counter_increment * len(result[i].masks)

                            args = ([results[i].cpu() for results in model_results], nodata_batch[i], bounds_batch[i], id_counter)
                            postproc_handler.put(args)

                    postproc_handler.sync()

            end = time.time()
            logger.debug(f"Pass ended in: {end - start_start} s.")

            postproc_handler.map()

            background = background_loader.get_background(planner.get_geotransform(), planner.region_size[0], planner.region_size[1], srs_wkt)
            postproc_handler.apply_background(background)

            postproc_handler.polygonize(planner.get_geotransform(), layer_info)
            postproc_handler.clear()

            region_counter += 1

            pbar_delineate.n = region_counter
            
            pbar_delineate.refresh()

    postproc_handler.dispose()
    logger.debug(f"Total time on creating dataloader: {total_dataloader_time} s.")

def postdelineation_merge(layer_info, filter_config):
    gpkg_path, layer_name = layer_info
    gpkg = ogr.Open(gpkg_path, 1)
    layer = gpkg.GetLayerByName(layer_name)

    MIN_AREA = filter_config["minimum_area_m2"]
    MIN_HOLE_AREA = filter_config["minimum_hole_area_m2"]

    max_id = 0
    try:
        layer.StartTransaction()

        # creating transform to equali-area projection
        src_srs = layer.GetSpatialRef()
        dst_src = osr.SpatialReference()
        dst_src.ImportFromEPSG(6933)

        transform = osr.CoordinateTransformation(src_srs, dst_src)

        # filtering polygons, and collect polygons what require merge
        field_parts = {}
        features_to_delete = []

        for feature in tqdm(layer, desc="Filtering", unit="poly"):
            fid = feature.GetFID()
            id = feature.GetField("id")
            bg = int(feature.GetField("bg"))
            if id > 0:
                feature.SetField("id", fid)
                layer.SetFeature(feature)
                continue

            orig_geom = feature.GetGeometryRef().Clone()

            if (id, bg) in field_parts:
                field_parts[(id, bg)].append(orig_geom)
            else:
                field_parts[(id, bg)] = [orig_geom]

            max_id = max(max_id, fid)
            features_to_delete.append(fid)
                
        # delete useless features
        for fid in tqdm(features_to_delete, desc="Deleting", unit="poly"):
            layer.DeleteFeature(fid)

        out_feat = ogr.Feature(layer.GetLayerDefn())
        # merge features and add them to the layer
        for key in tqdm(field_parts.keys(), desc="Merging", unit="poly"):
            id, bg = key
            cleaned_geoms = [g.Buffer(0) for g in field_parts[key] if g and not g.IsEmpty()]
            if not cleaned_geoms:
                logger.debug(f"Skipping id={id}: no valid geometries after cleaning")
                continue

            # Perform manual union (pairwise)
            merged = cleaned_geoms[0]
            for g in cleaned_geoms[1:]:
                merged = merged.Union(g)

            if merged is None or merged.IsEmpty():
                logger.debug(f"Skipping id={id}: union failed or returned empty")
                continue

            # Now decompose MultiPolygon into individual Polygon features
            geom_type = merged.GetGeometryType()
            if geom_type == ogr.wkbPolygon:
                parts = [merged]
            elif geom_type == ogr.wkbMultiPolygon:
                parts = [merged.GetGeometryRef(i).Clone() for i in range(merged.GetGeometryCount())]
            else:
                raise RuntimeError(f"Unexpected geometry type: {merged.GetGeometryName()}")

            # Create one feature per polygon part
            for part in parts:
                geom, area = PolygonizationWorker.remove_holes(part, MIN_HOLE_AREA, transform)
                if area < MIN_AREA:
                    continue

                out_feat.SetFID(-1)
                out_feat.SetGeometry(geom)
                out_feat.SetField("id", max_id + 1)
                out_feat.SetField("bg", bg)
                out_feat.SetField("area", float(area))
                layer.CreateFeature(out_feat)
                max_id += 1

        out_feat = None

        layer.CommitTransaction()
    except Exception as e:
        layer.RollbackTransaction()
        raise e
    
    gpkg.ExecuteSQL("VACUUM")

def warp_lclu(src, dst, sample_tiff, total_bounds, pixel_size, warp_options):
    temp_lclu_tiff_path = None

    mask_path = src
    if mask_path is not None:
        temp_lclu_tiff_path = dst

        if os.path.exists(dst):
            return temp_lclu_tiff_path

        sample_raster = gdal.Open(sample_tiff)
        target_proj = sample_raster.GetProjection()

        minx, miny, maxx, maxy = total_bounds

        cols = int(math.ceil((maxx - minx) / pixel_size[0]))
        rows = int(math.ceil((maxy - miny) / abs(pixel_size[1])))

        pbar = tqdm(total=100, desc="Warping LCLU", unit="%")

        def warping_progress_callback(complete, message, unknown):
            pbar.n = int(complete * 100)
            pbar.refresh()
            return 1

        # Perform the warp
        gdal.Warp(
            dst,
            mask_path,
            format='GTiff',
            dstSRS=target_proj,
            outputBounds=total_bounds,
            width=cols,
            height=rows,
            resampleAlg='nearest',
            creationOptions=warp_options,
            callback=warping_progress_callback,
        )

    return temp_lclu_tiff_path