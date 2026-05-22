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

# ---- Configuration ---------------------------------------------------------
BACKEND = "https://openeo.dataspace.copernicus.eu"
OUT_DIR = Path(__file__).resolve().parent / "test_outputs"

# The UDP process id as registered via generate_udp.py
PROCESS_ID = "esa_delineate_anything"

# AOI for testing (agricultural area in Belgium, ~20x20 km)
_W, _S, _E, _N = 4.95, 50.95, 5.13, 51.13
AOI = {
    "type": "Polygon",
    "coordinates": [[[_W, _S], [_E, _S], [_E, _N], [_W, _N], [_W, _S]]],
}
TEMPORAL = ["2024-04-01", "2024-09-30"]

# UDP parameters (use defaults by setting to None)
CONFIDENCE_THRESHOLD = 0.15
MASK_THRESHOLD = 0.2
MIN_AREA_PX = 10
MIN_HOLE_AREA_PX = 10
MAX_CLOUD_COVER = 75
# ----------------------------------------------------------------------------

print(f"Connecting to {BACKEND}...")
conn = openeo.connect(BACKEND)
conn.authenticate_oidc()

print(f"Calling UDP: {PROCESS_ID}")
cube = conn.datacube_from_process(
    process_id=PROCESS_ID,
    spatial_extent=AOI,
    temporal_extent=TEMPORAL,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    mask_threshold=MASK_THRESHOLD,
    min_area_px=MIN_AREA_PX,
    min_hole_area_px=MIN_HOLE_AREA_PX,
    max_cloud_cover=MAX_CLOUD_COVER,
)

title = "delineate_anything_udp_test"
out_format = "GTiff"

print(f"Submitting batch job: {title}")
job = cube.create_job(
    title=title,
    out_format=out_format,
)
job.start_and_wait()

print(f"Job finished: {job.job_id} (status: {job.status()})")



# %%