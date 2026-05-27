#%%
"""Test the registered Delineate-Anything UDP end-to-end.

This script:
  1. Connects to the CDSE openEO backend.
  2. Calls the registered user-defined process (UDP).
  3. Submits a batch job over a small AOI.
  4. Waits for completion and downloads the result.

Usage:
    python openeo_udp/tests/test_udp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import openeo
from openeo_udp.process_graph.delineate_onnx import DEFAULT_JOB_OPTIONS

# ---- Configuration ---------------------------------------------------------
BACKEND = "https://openeo.dataspace.copernicus.eu"
OUT_DIR = Path(__file__).resolve().parent / "test_outputs"

# The registered UDP process id (from generate_udp.py)
PROCESS_ID = "delineate_anything"

# AOI for testing (agricultural area in Belgium, ~20x20 km)
_W, _S, _E, _N = 4.95, 50.95, 5.13, 51.13
AOI = {
    "type": "Polygon",
    "coordinates": [[[_W, _S], [_E, _S], [_E, _N], [_W, _N], [_W, _S]]],
}
TEMPORAL = ["2024-04-01", "2024-09-30"]

PROCESSING_OPTIONS = {
    "confidence_threshold": 0.15,
    "mask_threshold": 0.2,
    "min_area_px": 10,
    "min_hole_area_px": 10,
}
MAX_CLOUD_COVER = 75
# ----------------------------------------------------------------------------

def main() -> None:
    print(f"Connecting to {BACKEND}...")
    conn = openeo.connect(BACKEND)
    conn.authenticate_oidc()

    print(f"Calling registered UDP: {PROCESS_ID}")
    cube = conn.datacube_from_process(
        process_id=PROCESS_ID,
        spatial_extent=AOI,
        temporal_extent=TEMPORAL,
        max_cloud_cover=MAX_CLOUD_COVER,
        processing_options=PROCESSING_OPTIONS,
    )

    title = "delineate_anything_udp_test"
    out_format = "GTiff"

    print(f"Submitting batch job: {title}")
    job = cube.create_job(
        title=title,
        out_format=out_format,
        job_options=DEFAULT_JOB_OPTIONS,
    )
    job.start_and_wait()

    print(f"Job finished: {job.job_id} (status: {job.status()})")



if __name__ == "__main__":
    main()



# %%