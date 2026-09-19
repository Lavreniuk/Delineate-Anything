# Author: Yevhenii Salii (https://github.com/15Litrov)

import multiprocessing
from osgeo import ogr
import traceback

class ReadWorker:
    @staticmethod
    def read_all_features_intersect_extent(fid_column, layer, extent, workers_queues):
        extent_geom = ReadWorker.make_extent_geom(extent)
        extent_geom.AssignSpatialReference(layer.GetSpatialRef())

        layer_defn = layer.GetLayerDefn()
        layer.SetSpatialFilter(extent_geom)

        counter = 0
        # --- Iterate features ---
        for feature in layer:
            fid = feature.GetField(fid_column)
            geom = feature.GetGeometryRef()
            if geom is None:
                continue

            if not geom.IsValid():
                geom = geom.MakeValid() if hasattr(geom, "MakeValid") else geom.Buffer(0)

            # Gather attributes
            fields = {}
            for i in range(layer_defn.GetFieldCount()):
                name = layer_defn.GetFieldDefn(i).GetName()
                fields[name] = feature.GetField(i)

            geom_wkb = geom.ExportToWkb()

            workers_queues[counter % len(workers_queues)].put((fid, geom_wkb, fields))
            counter += 1

        layer.SetSpatialFilter(None)

    @staticmethod
    def make_extent_geom(extent):
        minx, maxx, miny, maxy = extent
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(minx, miny)
        ring.AddPoint(maxx, miny)
        ring.AddPoint(maxx, maxy)
        ring.AddPoint(minx, maxy)
        ring.AddPoint(minx, miny)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        return poly