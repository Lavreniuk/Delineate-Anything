#%%


"""Test the Delineate-Anything openEO ONNX workflow end-to-end.

This script:
  1. Connects to the CDSE openEO backend.
  2. Builds the process graph (BAP composite → ONNX inference).
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
    build_delineate_onnx,
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
# ----------------------------------------------------------------------------

print(f"Connecting to {BACKEND}...")
conn = openeo.connect(BACKEND)
conn.authenticate_oidc()

print("Building process graph...")
cube = build_delineate_onnx(
    connection=conn,
    spatial_extent=AOI,
    temporal_extent=TEMPORAL,
)

print("Submitting batch job...")
job = cube.create_job(
    title="delineate_anything_onnx_test",
    out_format="GTiff",
    job_options=DEFAULT_JOB_OPTIONS,
)
job.start_and_wait()

print(f"Job finished: {job.job_id} (status: {job.status()})")

OUT_DIR.mkdir(parents=True, exist_ok=True)
results = job.get_results()
results.download_files(OUT_DIR)
print(f"Results downloaded to: {OUT_DIR}")
