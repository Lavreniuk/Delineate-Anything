#%%


"""Test the Delineate-Anything openEO ONNX workflow end-to-end.

This script:
  1. Connects to the CDSE openEO backend.
  2. Builds the full process graph (BAP → inference → post-processing).
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

from openeo_udp.process_graph.delineate_onnx import (
    build_bap_only,
    build_delineate_onnx,
    build_delineate_full,
    DEFAULT_JOB_OPTIONS,
)

# ---- Configuration ---------------------------------------------------------
BACKEND = "https://openeo.dataspace.copernicus.eu"
OUT_DIR = Path(__file__).resolve().parent / "test_outputs"

# Small AOI for testing (agricultural area in Belgium)
AOI = {
    "west": 5.0,
    "south": 51.0,
    "east": 5.05,
    "north": 51.05,
    "crs": "EPSG:4326",
}
TEMPORAL = ["2024-04-01", "2024-09-30"]

# Set to True to run the full pipeline (inference + post-processing)
# Set to False to run inference only
RUN_FULL_PIPELINE = True
# Set to "bap" to only export the BAP composite (input to UDF) as NetCDF
# Options: "bap", "inference", "full"
MODE = "bap"
# ----------------------------------------------------------------------------

print(f"Connecting to {BACKEND}...")
conn = openeo.connect(BACKEND)
conn.authenticate_oidc()

print("Building process graph...")
if MODE == "bap":
    print("  Mode: BAP composite only")
    cube = build_bap_only(
        connection=conn,
        spatial_extent=AOI,
        temporal_extent=TEMPORAL,
    )
    title = "delineate_anything_bap_test"
    out_format = "NetCDF"
elif MODE == "full":
    print("  Mode: full pipeline (inference + post-processing)")
    cube = build_delineate_full(
        connection=conn,
        spatial_extent=AOI,
        temporal_extent=TEMPORAL,
    )
    title = "delineate_anything_full_test"
    out_format = "GTiff"
else:
    print("  Mode: inference only")
    cube = build_delineate_onnx(
        connection=conn,
        spatial_extent=AOI,
        temporal_extent=TEMPORAL,
    )
    title = "delineate_anything_onnx_test"
    out_format = "GTiff"

title = "delineate_anything_full_test" if RUN_FULL_PIPELINE else "delineate_anything_onnx_test"
print(f"Submitting batch job: {title}")
job = cube.create_job(
    title=title,
    out_format=out_format,
    job_options=DEFAULT_JOB_OPTIONS,
)
job.start_and_wait()

print(f"Job finished: {job.job_id} (status: {job.status()})")


