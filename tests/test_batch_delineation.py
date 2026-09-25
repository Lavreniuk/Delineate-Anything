from argparse import Namespace
from pathlib import Path
import sys

import yaml
from osgeo import ogr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from delineate import batch_routine


EXPECTED_POLYGON_COUNT = 678


def test_batch_sample_delineate(tmp_path):
    batch_config = yaml.safe_load((REPO_ROOT / "batch_sample.yaml").read_text())
    batch_config["base_config"] = str(REPO_ROOT / batch_config["base_config"])
    batch_config["data_root"] = str(REPO_ROOT / batch_config["data_root"])
    batch_config["mask_root"] = str(REPO_ROOT / batch_config["mask_root"])
    batch_config["output_root"] = str(tmp_path / "delineated")
    batch_config["temp_root"] = str(tmp_path / "temp")

    batch_config_path = tmp_path / "batch_sample.yaml"
    batch_config_path.write_text(yaml.safe_dump(batch_config))

    output_paths = batch_routine(
        Namespace(batch_config=str(batch_config_path), verbose=False)
    )

    assert len(output_paths) == 1
    dataset = ogr.Open(output_paths[0])
    assert dataset is not None
    layer = dataset.GetLayerByName("fields")
    assert layer is not None
    assert layer.GetFeatureCount() == EXPECTED_POLYGON_COUNT