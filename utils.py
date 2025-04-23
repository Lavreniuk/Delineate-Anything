import os
import time
import numpy as np
import cv2
from osgeo import gdal, osr, ogr
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "create_tiff_with_same_bounds",
    "create_geopackage_with_same_projection",
    "get_biggest_component_raster",
    "rasterize",
    "find_edge_mapping",
    "cleanup_geopackage",
    "get_extent",
    "merge_extents",
    "create_empty_tiff",
    "get_pixel_offset"
]


def create_tiff_with_same_bounds(dst_path, src_ds, type, scale_factor, offset_x, offset_y):
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize

    scaled_cols = cols * scale_factor + offset_x
    scaled_rows = rows * scale_factor + offset_y

    scaled_geotransform = (
        geotransform[0],
        geotransform[1] / scale_factor,
        geotransform[2],
        geotransform[3],
        geotransform[4],
        geotransform[5] / scale_factor
    )

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(dst_path, scaled_cols, scaled_rows, 1, type, options=["BIGTIFF=YES", "COMPRESS=DEFLATE"])
    dst_ds.SetGeoTransform(scaled_geotransform)
    dst_ds.SetProjection(projection)

    return dst_ds


def create_geopackage_with_same_projection(dst_path, layer_name, src_ds, override_if_exists):
    if os.path.exists(dst_path):
        if override_if_exists:
            os.remove(dst_path)
        else:
            raise RuntimeError(f"ERROR: {dst_path} already exists!")

    proj_wkt = src_ds.GetProjection()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj_wkt)

    driver = ogr.GetDriverByName("GPKG")
    gpkg_ds = driver.CreateDataSource(dst_path)
    if gpkg_ds is None:
        raise RuntimeError(f"ERROR: Could not create GeoPackage: {dst_path}")

    layer = gpkg_ds.CreateLayer(layer_name, srs=spatial_ref, geom_type=ogr.wkbPolygon)
    return gpkg_ds, layer

def get_biggest_component_raster(mask):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype="uint8")
    buff = mask.astype("uint8")
    buff = cv2.morphologyEx(buff, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    contours, _ = cv2.findContours(buff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.squeeze(1) for c in contours]
    areas = np.array([cv2.contourArea(c) for c in contours])

    buff[:, :] = 0
    if len(areas) == 0:
        return buff

    max_contour = contours[np.argmax(areas)]
    buff = cv2.fillPoly(buff, [max_contour.astype("int32")], 1, cv2.LINE_4)
    buff = cv2.morphologyEx(buff, cv2.MORPH_DILATE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return buff


def rasterize(result, id_counter, pixel_area_threshold, remaining_area_threshold, nodata_mask):
    dnp = result.masks.data.cpu().numpy()

    init_id = id_counter
    instance_map = np.zeros(dnp.shape[1:], dtype="uint32")
    conf_map = np.zeros(dnp.shape[1:], dtype="float32")
    init_area = [1] * (dnp.shape[0] + 1)
    areas = {}

    for i in range(1, dnp.shape[0] + 1):
        component = get_biggest_component_raster(dnp[-i])
        dnp[-i] = component
        area = np.sum(component & nodata_mask)

        id_counter += 1
        if area < pixel_area_threshold:
            continue

        conf = float(result.boxes[-i].conf[0].cpu())
        instance_mask = component > 0
        instance_map[instance_mask] = id_counter
        conf_map[instance_mask] = conf
        init_area[i] = area
        areas[id_counter] = area

    survivors = set()
    unique = np.unique(instance_map, return_counts=True)

    for i in range(len(unique[0])):
        _id = unique[0][i]
        if _id == 0:
            continue
        count = unique[1][i]
        if count < pixel_area_threshold or (count / float(areas[_id])) < remaining_area_threshold:
            continue
        survivors.add(_id)

    instance_map[:, :] = 0
    conf_map[:, :] = 0

    for i in range(1, dnp.shape[0] + 1):
        _id = init_id + i
        if _id not in survivors:
            continue

        conf = float(result.boxes[-i].conf[0].cpu())
        instance_mask = dnp[-i] > 0
        instance_map[instance_mask] = _id
        conf_map[instance_mask] = conf

    return instance_map, conf_map, id_counter


def find_edge_mapping(current, new, dst, merge_relative_area_threshold, merge_asymetric_pixel_area_threshold, merge_asymetric_relative_area_threshold):
    uc = np.unique(current, return_counts=True)
    un = np.unique(new, return_counts=True)
    ui = np.unique((current.astype("uint64") << 32) | new.astype("uint64"), return_counts=True)

    if uc[0].shape[0] == 0 or (uc[0][0] == 0 and uc[0].shape[0] == 1) or ui[0].shape[0] == 0 or (ui[0][0] == 0 and ui[0].shape[0] == 1):
        return dst

    for i in range(ui[0].shape[0]):
        key = np.uint64(ui[0][i])
        key_current = key >> np.uint64(32)
        key_new = key & np.uint64(0xFFFFFFFF)

        if key_current == 0 or key_new == 0:
            continue

        uc_idx = np.argwhere(uc[0] == key_current)
        un_idx = np.argwhere(un[0] == key_new)

        if uc_idx.shape[0] == 0 or un_idx.shape[0] == 0:
            continue

        rel_area_current = float(ui[1][i]) / uc[1][uc_idx[0]]
        rel_area_new = float(ui[1][i]) / un[1][un_idx[0]]

        if (rel_area_current > merge_relative_area_threshold and rel_area_new > merge_relative_area_threshold) or \
           (uc[1][uc_idx[0]] > merge_asymetric_pixel_area_threshold and rel_area_current > merge_asymetric_relative_area_threshold) or \
           (un[1][un_idx[0]] > merge_asymetric_pixel_area_threshold and rel_area_new > merge_asymetric_relative_area_threshold):
            dst[int(key_new)] = dst.get(int(key_new), []) + [int(key_current)]

    return dst


def cleanup_geopackage(layer, area_threshold):
    logger.info("Starting geometry cleanup...")
    t_start = time.time()
    layer.StartTransaction()

    features_to_delete = []
    for i, feature in enumerate(layer):
        geom = feature.GetGeometryRef()
        geom_type = geom.GetGeometryType()

        if geom_type == ogr.wkbPolygon:
            outer = geom.GetGeometryRef(0).Clone()
            cleaned_geom = ogr.Geometry(ogr.wkbPolygon)
            cleaned_geom.AddGeometry(outer)
        elif geom_type == ogr.wkbMultiPolygon:
            cleaned_geom = ogr.Geometry(ogr.wkbMultiPolygon)
            for j in range(geom.GetGeometryCount()):
                poly = geom.GetGeometryRef(j)
                outer = poly.GetGeometryRef(0).Clone()
                new_poly = ogr.Geometry(ogr.wkbPolygon)
                new_poly.AddGeometry(outer)
                cleaned_geom.AddGeometry(new_poly)
        else:
            continue

        area = cleaned_geom.GetArea()
        if area < area_threshold:
            features_to_delete.append(feature.GetFID())
        else:
            feature.SetGeometry(cleaned_geom)
            layer.SetFeature(feature)

    logger.info(f"Cleaning complete in {time.time() - t_start:.2f} s.")

    t_start = time.time()
    for i, fid in enumerate(features_to_delete):
        layer.DeleteFeature(fid)
    logger.info(f"Deletion {len(features_to_delete)} small features complete in {time.time() - t_start:.2f} s.")
    layer.CommitTransaction()


def get_extent(ds):
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    minx = gt[0]
    maxx = gt[0] + cols * gt[1]
    miny = gt[3] + rows * gt[5]
    maxy = gt[3]
    return minx, miny, maxx, maxy


def merge_extents(extents):
    minx = min(e[0] for e in extents)
    miny = min(e[1] for e in extents)
    maxx = max(e[2] for e in extents)
    maxy = max(e[3] for e in extents)
    return minx, miny, maxx, maxy


def create_empty_tiff(output_path, extent, projection, pixel_size, dtype):
    minx, miny, maxx, maxy = extent
    res = pixel_size
    cols = int((maxx - minx) / res) + 1
    rows = int((maxy - miny) / res) + 1

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, cols, rows, 1, dtype, options=["BIGTIFF=YES", "COMPRESS=DEFLATE"])
    gt = (minx, res, 0, maxy, 0, -res)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(projection)

    return out_ds


def get_pixel_offset(ds1, ds2):
    gt1 = ds1.GetGeoTransform()
    gt2 = ds2.GetGeoTransform()

    pixel_width = gt1[1]
    pixel_height = gt1[5]

    offset_x = (gt2[0] - gt1[0]) / pixel_width
    offset_y = (gt2[3] - gt1[3]) / pixel_height

    return int(round(offset_x)), int(round(offset_y))
