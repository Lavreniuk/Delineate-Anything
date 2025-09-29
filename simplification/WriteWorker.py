import multiprocessing
from osgeo import ogr

from tqdm import tqdm
import traceback

class WriteWorker(multiprocessing.Process):
    MODE_TERMINATE = 0

    def __init__(self, dst_gpkg, dst_layer, input_queue, total, feature_counter):
        super().__init__(daemon=False)

        self.dst_qpkg = dst_gpkg
        self.dst_layer = dst_layer
        self.input_queue = input_queue
        self.total = total

        self.writted_ids = set()
        self.started_event = multiprocessing.Event()

        self.feature_counter = feature_counter

    def run(self):
        self.started_event.set()

        dst = ogr.Open(self.dst_qpkg, 1)
        layer = dst.GetLayerByName(self.dst_layer)

        if self.total:
            pbar = tqdm(total=self.total, desc="Writing features", position=0, unit="poly")
        else:
            pbar = tqdm(desc="Writing features", position=0, unit="poly")

        counter = 0
        layer.StartTransaction()
        while True:
            try:
                command = self.input_queue.get(timeout=0.1)
            except:
                continue

            if command == WriteWorker.MODE_TERMINATE or command is None:
                break
            
            fid, wkb, fields = command

            if fid == -1:
                self.feature_counter.value += 1
                continue

            try:
                geom = ogr.CreateGeometryFromWkb(wkb)

                # create new feature with same schema
                defn = layer.GetLayerDefn()
                feature = ogr.Feature(defn)
                feature.SetGeometry(geom)

                # copy fields
                for i in range(feature.GetFieldCount()):
                    name = defn.GetFieldDefn(i).GetName()
                    feature.SetField(i, fields[name])

                layer.CreateFeature(feature)
                feature = None
                self.writted_ids.add(fid)
                self.feature_counter.value += 1
                counter += 1
                inc = len(self.writted_ids) - pbar.n
                pbar.update(inc)

                if counter >= 1000:
                    counter = 0
                    layer.CommitTransaction()
                    layer.StartTransaction()
            except:
                traceback.print_exc()

        layer.CommitTransaction()