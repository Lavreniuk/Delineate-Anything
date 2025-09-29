from simplification import simplify
import time

from argparse import ArgumentParser
import yaml
from pathlib import Path

from osgeo import gdal
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

def main():
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", dest="config",
                help="Configuration of execution (*.yaml).")
    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(Path(args.config).read_text())
        simplify.simplify(config)
    else:
        parser.print_help()

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    print("Simplification finished in", end - start, "s.")