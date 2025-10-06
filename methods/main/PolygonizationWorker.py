from rasterio import features
from shapely.geometry import shape
from shapely.affinity import affine_transform as shapely_affine_transform
from shapely.geometry import shape, box
from affine import Affine

from osgeo import osr, ogr

import numpy as np

class PolygonizationWorker:
    @staticmethod
    def polygonize(queue, shm_instances, raster_shape, geotransform, config, srs_wkt, region_begin, region_end):
        instances = np.ndarray(shape=raster_shape, dtype=np.int32, buffer=shm_instances.buf)

        array = instances[region_begin[0]:region_end[0], region_begin[1]:region_end[1]]

        srs = osr.SpatialReference()
        srs.ImportFromWkt(srs_wkt)

        min_area_m2 = config["minimum_area_m2"]
        min_part_area_m2 = config["minimum_part_area_m2"]
        min_hole_area_m2 = config["minimum_hole_area_m2"]

        min_background_field_area_m2 = config["minimum_background_field_area_m2"]
        min_background_field_hole_area_m2 = config["minimum_background_field_hole_area_m2"]

        middleground_offset = config["middleground_offset"]
        min_middleground_field_area_m2 = config["minimum_middleground_field_area_m2"]
        min_middleground_field_hole_area_m2 = config["minimum_middleground_field_hole_area_m2"]

        gt = geotransform
        affine_transform = Affine.from_gdal(*gt)
        shapes_generator = features.shapes(array, mask=(array < 0) | (array > 1))
        shapes_generator = list(shapes_generator) 
        
        equal_area_srs = osr.SpatialReference()
        equal_area_srs.ImportFromEPSG(6933)
        coord_transform = osr.CoordinateTransformation(srs, equal_area_srs)

        height, width = array.shape
        image_bbox = box(0, 0, width, height)

        a, b, c, d, e, f = affine_transform.a, affine_transform.b, affine_transform.c, affine_transform.d, affine_transform.e, affine_transform.f
        shapely_params = (a, b, d, e, c, f)

        # receiving polygons in pixel coorinates
        for geom_json, value in shapes_generator:
            if value >= 0 and value < 2:
                continue

            geom = shape(geom_json)
            poly_bbox = box(*geom.bounds)
            touches_edge = poly_bbox.intersects(image_bbox.boundary)

            # converting it to dst crs
            geom_geo = shapely_affine_transform(geom, shapely_params)
            if geom_geo.is_empty or not geom_geo.is_valid:
                continue

            wkb = geom_geo.wkb
            if not wkb:
                continue
        
            ogr_geom = ogr.CreateGeometryFromWkb(wkb)
            if ogr_geom is None or ogr_geom.IsEmpty():
                continue
            
            # removing holes what is too small to be realistically useful
            isBackground = value < 0
            isMiddleground = isBackground and (value > -middleground_offset)
            if not isBackground:
                ogr_geom, area = PolygonizationWorker.remove_holes(ogr_geom, min_hole_area_m2, coord_transform)
            elif not isMiddleground:
                ogr_geom, area = PolygonizationWorker.remove_holes(ogr_geom, min_background_field_hole_area_m2, coord_transform)
            else:
                ogr_geom, area = PolygonizationWorker.remove_holes(ogr_geom, min_middleground_field_hole_area_m2, coord_transform)

            # if polygon is too small -> skip it
            if (not isBackground and ((touches_edge and area < min_part_area_m2) or (not touches_edge and area < min_area_m2))) or\
               (isBackground and not isMiddleground and (area < min_background_field_area_m2)) or\
                (isMiddleground and (area < min_middleground_field_area_m2)):
               continue
            
            # creating polygon id and mark it with negative sign if we want to merge it with others in postdelineation merge
            result_id = -int(value) if touches_edge else int(value)
            if isBackground:
                result_id = -result_id

            queue.put((ogr_geom.ExportToWkb(), float(area), result_id, int(isBackground)))

    @staticmethod
    def remove_holes(geom, min_hole_area, transform):
        area_geom = geom.Clone()
        area_geom.Transform(transform)

        parts_count = area_geom.GetGeometryCount()
        parts_area = np.empty((parts_count), dtype=np.float32)
        for i in range(parts_count):
            parts_area[i] = area_geom.GetGeometryRef(i).GetArea()

        total_area = parts_area[0]

        if parts_count == 1:
            return geom, total_area
        
        rings = []
        for i in range(1, parts_count):
            if parts_area[i] < min_hole_area:
                continue

            total_area -= parts_area[i]
            rings.append(geom.GetGeometryRef(i))

        new_poly = ogr.Geometry(ogr.wkbPolygon)
        new_poly.AddGeometry(geom.GetGeometryRef(0))
        for ring in rings:
            new_poly.AddGeometry(ring)

        return new_poly, total_area