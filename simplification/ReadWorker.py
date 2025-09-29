import multiprocessing
from osgeo import ogr

class ReadWorker(multiprocessing.Process):
    MODE_TERMINATE = 0
    MODE_READ_ALL = 1
    MODE_READ_NONSIMPLIFIED = 2

    def __init__(self, src_gpkg, src_layer, fid_column, output_queue, shared_set, start_flag, done_flag, counter_value):
        super().__init__(daemon=False)

        self.input_queue = multiprocessing.JoinableQueue()
        self.started_event = multiprocessing.Event()
        self.output_queue = output_queue
        self.fid_column = fid_column

        self.shared_set = shared_set

        self.src_gpkg = src_gpkg
        self.src_layer = src_layer

        self.out_of_extent =  multiprocessing.Event()

        self.start_flag = start_flag
        self.done_flag = done_flag
        self.counter_value = counter_value
        self.local_counter = 0

    def run(self):
        self.started_event.set()

        src = ogr.Open(self.src_gpkg, 0)
        layer = src.GetLayerByName(self.src_layer)

        while True:
            self.start_flag.wait()
            self.start_flag.clear()

            try:
                command, args = self.input_queue.get(timeout=1)
            except:
                continue
            
            if command == ReadWorker.MODE_TERMINATE:
                self.done_flag.set()
                return
            elif command == ReadWorker.MODE_READ_ALL:
                self.read_all_features_intersect_extent(layer, args)
            elif command == ReadWorker.MODE_READ_NONSIMPLIFIED:
                self.read_all_features_intersect_extent(layer, args)

            self.done_flag.set()

    def read_all_features_intersect_extent(self, layer, extent):
        self.local_counter = 0
        extent_geom = ReadWorker.make_extent_geom(extent)

        # spatial filter narrows down candidates first
        layer.SetSpatialFilter(extent_geom)

        for feature in layer:
            fid = feature.GetField(self.fid_column)
            geom = feature.GetGeometryRef()
            if geom is None:
                continue

            # clip geometry to extent
            clipped_geom = geom.Intersection(extent_geom)
            if clipped_geom is None or clipped_geom.IsEmpty():
                continue

            defn = feature.GetDefnRef()
            fields = {}
            for i in range(defn.GetFieldCount()):
                name = defn.GetFieldDefn(i).GetName()
                fields[name] = feature.GetField(i)

            gtype = clipped_geom.GetGeometryType()

            if gtype == ogr.wkbMultiPolygon or gtype == ogr.wkbMultiPolygon25D:
                # Split multipolygon into individual polygons
                for i in range(clipped_geom.GetGeometryCount()):
                    poly = clipped_geom.GetGeometryRef(i)
                    if poly is None or poly.IsEmpty():
                        continue

                    geom_wkb = poly.ExportToWkb()

                    item = (fid, geom_wkb, fields)
                    self.output_queue.put(item)
                    self.local_counter += 1

            else:
                geom_wkb = clipped_geom.ExportToWkb()

                item = (fid, geom_wkb, fields)
                self.output_queue.put(item)
                self.local_counter += 1

        layer.SetSpatialFilter(None)
        self.counter_value.value = self.local_counter



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