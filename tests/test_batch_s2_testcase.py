from argparse import Namespace
import os
from pathlib import Path
import sys

import pytest
import yaml
from osgeo import ogr


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = Path(__file__).resolve().parent / "testcases" / "2026_s2_test"
sys.path.insert(0, str(REPO_ROOT))

from delineate import batch_routine


def test_2026_s2_tile_edge_case(tmp_path):
    batch = yaml.safe_load((CASE_ROOT / "batch_s2_test.yaml").read_text())
    batch["base_config"] = str(CASE_ROOT / batch["base_config"])
    batch["data_root"] = str(CASE_ROOT / batch["data_root"])
    batch["mask_root"] = str(CASE_ROOT / batch["mask_root"])
    batch["temp_root"] = str(tmp_path / "temp")
    batch["output_root"] = os.environ.get(
        "S2_TEST_OUTPUT_ROOT", str(tmp_path / "output")
    )

    batch_path = tmp_path / "batch_s2_test.yaml"
    batch_path.write_text(yaml.safe_dump(batch))
    output_paths = batch_routine(
        Namespace(batch_config=str(batch_path), verbose=False)
    )

    assert len(output_paths) == 1
    dataset = ogr.Open(output_paths[0])
    assert dataset is not None
    layer = dataset.GetLayerByName("fields")
    assert layer is not None
    assert layer.GetFeatureCount() > 0