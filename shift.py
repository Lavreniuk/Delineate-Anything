import os    
from argparse import ArgumentParser

from osgeo import gdal, ogr
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm

def shift_single_feature(data):
    fid, wkt, dx, dy = data
    geom = ogr.CreateGeometryFromWkt(wkt)

    for i in range(geom.GetGeometryCount()):
        ring = geom.GetGeometryRef(i)
        for j in range(ring.GetPointCount()):
            x, y, z = ring.GetPoint(j)
            ring.SetPoint(j, x + dx, y + dy, z)

    return fid, geom.ExportToWkt()

class GeometryWorker(Process):
    def __init__(self, in_queue, out_queue, dx, dy):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.dx = dx
        self.dy = dy

    def run(self):
        while True:
            item = self.in_queue.get()
            if item is None:
                break  # Sentinel to shut down
            fid, wkt, field_vals = item
            geom = ogr.CreateGeometryFromWkt(wkt)
            for i in range(geom.GetGeometryCount()):
                ring = geom.GetGeometryRef(i)
                for j in range(ring.GetPointCount()):
                    x, y, z = ring.GetPoint(j)
                    ring.SetPoint(j, x + self.dx, y + self.dy, z)
            self.out_queue.put((fid, geom.ExportToWkt(), field_vals))

def shifter(input_path, output_path, dx, dy):
    # Open input layer
    driver = ogr.GetDriverByName("GPKG")
    src_ds = driver.Open(input_path, 0)
    src_layer = src_ds.GetLayer()
    layer_name = src_layer.GetName()

    # Create output layer
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    dst_ds = driver.CreateDataSource(output_path)

    dst_layer = dst_ds.CreateLayer(layer_name, src_layer.GetSpatialRef(), src_layer.GetGeomType())
    layer_def = src_layer.GetLayerDefn()
    for i in range(layer_def.GetFieldCount()):
        dst_layer.CreateField(layer_def.GetFieldDefn(i))

    # Queues
    in_queue = Queue(maxsize=8096)
    out_queue = Queue()

    # Start workers
    workers = [GeometryWorker(in_queue, out_queue, dx, dy) for _ in range(cpu_count())]
    for w in workers:
        w.start()

    # Send features to workers
    total = src_layer.GetFeatureCount()
    dst_layer.StartTransaction()

    pbar = tqdm(total=total, desc="Shifting", unit="feature")
    sent = 0
    written = 0
    fid_list = []
    for feature in src_layer:
        geom = feature.GetGeometryRef()
        if not geom:
            continue
        fid = feature.GetFID()
        wkt = geom.ExportToWkt()
        fields = [feature.GetField(i) for i in range(layer_def.GetFieldCount())]
        in_queue.put((fid, wkt, fields))
        fid_list.append(fid)
        sent += 1

        # Write back as workers finish
        while out_queue.qsize() > 0:
            fid_out, new_wkt, field_vals = out_queue.get()
            out_feature = ogr.Feature(dst_layer.GetLayerDefn())
            out_feature.SetGeometry(ogr.CreateGeometryFromWkt(new_wkt))
            for i, val in enumerate(field_vals):
                out_feature.SetField(i, val)
            dst_layer.CreateFeature(out_feature)
            out_feature = None
            written += 1
            pbar.update(1)

    # Signal termination
    for _ in workers:
        in_queue.put(None)

    # Drain remaining results
    while written < sent:
        _, new_wkt, field_vals = out_queue.get()
        out_feature = ogr.Feature(dst_layer.GetLayerDefn())
        out_feature.SetGeometry(ogr.CreateGeometryFromWkt(new_wkt))
        for i, val in enumerate(field_vals):
            out_feature.SetField(i, val)
        dst_layer.CreateFeature(out_feature)
        out_feature = None
        written += 1
        pbar.update(1)

    pbar.close()
    dst_layer.CommitTransaction()

    # Cleanup
    for w in workers:
        w.join()
    src_ds = None
    dst_ds = None
    print(f"Done. Written to: {output_path}")


def main():
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", dest="input", default=None,
                help="Src file.")
    
    parser.add_argument("-o", "--output", dest="output", default=None,
                help="Dst file.")
    
    parser.add_argument("-s", "--sample", dest="sample", default=None,
                help="Raster file to determine pixel size. If present shift units will be in pixels.")
    
    parser.add_argument("-x", "--shift_x", dest="shift_x", type=float, default=0,
                help="How many units of shift to apply on X axis.")
    
    parser.add_argument("-y", "--shift_y", dest="shift_y", type=float, default=0,
                help="How many units of shift to apply on Y axis.")
    
    args = parser.parse_args()  

    if not args.input or not args.output or not args.shift_x or not args.shift_y:
        parser.print_help()

    pixel_size = (1.0, 1.0)
    if args.sample:
        ds = gdal.Open(args.sample)
        gt = ds.GetGeoTransform()
        pixel_size = (gt[1], gt[5])

    shift = (pixel_size[0] * args.shift_x, pixel_size[1] * args.shift_y)
    shifter(args.input, args.output, shift[0], shift[1])

if __name__ == "__main__":
    main()