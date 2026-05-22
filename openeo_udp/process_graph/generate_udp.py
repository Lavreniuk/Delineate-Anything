#%%
"""Generate Delineate-Anything UDP JSON and optionally register it.

Edit the constants below, then run:
    python openeo_udp/process_graph/generate_udp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openeo
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict

from openeo_udp.process_graph.delineate_onnx import (
    DEFAULT_JOB_OPTIONS,
    build_delineate_full,
)

# ---- edit here -------------------------------------------------------------
BACKEND = "https://openeo.dataspace.copernicus.eu"
PROCESS_ID = "esa_delineate_anything"
OUTPUT_JSON = Path(__file__).resolve().parent / "delineate_anything_udp.json"
REGISTER = True
# ---------------------------------------------------------------------------

conn = openeo.connect(BACKEND)
conn.authenticate_oidc()

spatial_extent = Parameter(
    name="spatial_extent",
    description="GeoJSON geometry (Polygon) defining the area of interest.",
    schema={"type": "object"},
)
temporal_extent = Parameter(
    name="temporal_extent",
    description="Date range [start, end] (ISO-8601).",
    schema={
        "type": "array",
        "items": {"type": "string"},
        "minItems": 2,
        "maxItems": 2,
    },
)
confidence_threshold = Parameter(
    name="confidence_threshold",
    description="YOLO detection confidence threshold (0-1). Lower = more detections.",
    schema={"type": "number"},
    default=0.15,
)
mask_threshold = Parameter(
    name="mask_threshold",
    description="Binarisation threshold for mask probability (0-1). Lower = more mask area.",
    schema={"type": "number"},
    default=0.2,
)
min_area_px = Parameter(
    name="min_area_px",
    description="Minimum field area in pixels. Smaller fields are removed.",
    schema={"type": "integer"},
    default=10,
)
min_hole_area_px = Parameter(
    name="min_hole_area_px",
    description="Minimum hole area in pixels to preserve. Smaller holes are filled.",
    schema={"type": "integer"},
    default=10,
)
max_cloud_cover = Parameter(
    name="max_cloud_cover",
    description="Maximum cloud cover percentage for BAP input scenes.",
    schema={"type": "integer"},
    default=75,
)

cube = build_delineate_full(
    connection=conn,
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    confidence_threshold=confidence_threshold,
    mask_threshold=mask_threshold,
    min_area_px=min_area_px,
    min_hole_area_px=min_hole_area_px,
    max_cloud_cover=max_cloud_cover,
)

udp = build_process_dict(
    process_graph=cube,
    process_id=PROCESS_ID,
    summary="Delineate-Anything field boundary detection (ONNX)",
    description=(
        "BAP RGB composite → YOLO-seg ONNX inference → post-processing. "
        "Returns 3 bands: mask_probability (float), binary_mask (0/1), "
        "instances (integer field labels)."
    ),
    parameters=[
        spatial_extent,
        temporal_extent,
        confidence_threshold,
        mask_threshold,
        min_area_px,
        min_hole_area_px,
        max_cloud_cover,
    ],
    default_job_options=DEFAULT_JOB_OPTIONS,
)

OUTPUT_JSON.write_text(json.dumps(udp, indent=2), encoding="utf-8")
print(f"Wrote {OUTPUT_JSON}")

if REGISTER:
    conn.save_user_defined_process(
        user_defined_process_id=PROCESS_ID,
        process_graph=udp["process_graph"],
        parameters=udp.get("parameters", []),
        summary=udp.get("summary"),
        description=udp.get("description"),
    )
    print(f"Registered UDP: {PROCESS_ID}")

# %%
