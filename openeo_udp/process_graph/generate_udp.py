#%%

"""Generate a UDP JSON for the full Delineate-Anything workflow.

The UDP wraps :func:`build_delineate_onnx` (BAP composite + apply_neighborhood
with the ONNX UDF) and exposes two runtime parameters:

- ``geometry`` (GeoJSON geometry)
- ``temporal_extent`` (``[start, end]`` ISO date strings)

Run the generated JSON on a backend with::

    conn.datacube_from_json(
        "openeo_udp/process_graph/delineate_anything_udp.json",
        parameters={"geometry": geom, "temporal_extent": ["2024-05-01", "2024-08-31"]},
    )
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import openeo
from openeo.api.process import Parameter

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openeo_udp.tests.test_udf_cdse import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    DEFAULT_JOB_OPTIONS,
    DEFAULT_WEIGHTS_URL,
    IOU_THRESHOLD,
    build_delineate_onnx,
)

DEFAULT_PROCESS_ID = "delineate_anything"
DEFAULT_BACKEND = "https://openeo.dataspace.copernicus.eu"
OUTPUT_PATH = Path(__file__).with_name(f"{DEFAULT_PROCESS_ID}_udp.json")

DEFAULT_PALOUSE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[
        [-117.0635, 46.7150],
        [-116.9325, 46.7150],
        [-116.9325, 46.8050],
        [-117.0635, 46.8050],
        [-117.0635, 46.7150],
    ]],
}


DEFAULT_DESCRIPTION = """\
This process targets one core problem in agricultural remote sensing: turning broad, \
repeated satellite coverage into field-level structure that can be compared across time \
and places. Instead of emphasizing pixel-wise texture alone, it emphasizes parcel \
coherence, boundary continuity, and map readability.

In practical terms, this helps users move from "imagery" to "field units" that are easier \
to interpret for crop monitoring, stratified sampling, landscape diagnostics, and \
downstream analytics where parcel identity matters.

The underlying model is Delineate Anything v2, a resolution-agnostic global foundation model for agricultural \
field boundary detection from satellite imagery (Lavreniuk et al., ECCV 2026, \
arXiv:2607.19069). Built on a YOLO instance-segmentation backbone and trained on \
FBIS-73M — a 73-million-instance dataset spanning 61 countries and resolutions from \
0.25 m to 10 m — it is designed to map field structure consistently across diverse \
landscapes, seasons, and management regimes.

This process is organized as a four-stage pipeline from reflectance to parcel geometry:

1) Sentinel-2 preparation. A Best-Available-Pixel (BAP) Sentinel-2 L2A RGB composite is \
assembled for the requested geometry and temporal interval.
2) Tile-wise instance segmentation. The Delineate Anything v2 ONNX model runs in a \
Python UDF on 512x512 tiles with overlap to preserve boundary continuity near tile edges.
3) Raster-to-vector conversion. Instance labels are polygonised with `raster_to_vector`, \
yielding a vector cube where each polygon represents one detected field candidate.
4) Seam reconciliation and ordering. A second vector UDF merges polygons split by chunk \
boundaries and assigns `field_id` from north to south, west to east for stable indexing.

Operationally, inference keeps a 384 px core per tile and uses 64 px context overlap on \
each side. The raster UDF encodes chunk identity, enabling deterministic seam \
reconciliation in the vector UDF. Output is intended for direct export as GeoJSON or \
GeoParquet.

Known limitation: if two truly different fields share exactly the same border along a \
chunk seam, they may be merged during seam reconciliation.
"""


def build_udp(
    *,
    process_id: str = DEFAULT_PROCESS_ID,
    backend: str = DEFAULT_BACKEND,
    weights_url: str = DEFAULT_WEIGHTS_URL,
) -> dict:
    """Build a UDP dict wrapping the Delineate-Anything workflow."""
    geometry = Parameter.geojson(
        name="geometry",
        description="GeoJSON geometry defining the area of interest.",
    )
    temporal_extent = Parameter.temporal_interval(
        name="temporal_extent",
        description="Temporal interval as [start, end] ISO-8601 date strings.",
    )
    iou_threshold = Parameter.number(
        name="iou_threshold",
        description="NMS IoU threshold. Lower values suppress more overlapping detections.",
        default=IOU_THRESHOLD,
    )
    confidence_threshold = Parameter.number(
        name="confidence_threshold",
        description="Minimum detection confidence.",
        default=CONFIDENCE_THRESHOLD,
    )

    conn = openeo.connect(backend)
    cube = build_delineate_onnx(
        connection=conn,
        geometry=geometry,
        temporal_extent=temporal_extent,
        weights_url=weights_url,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )

    geometry_dict = geometry.to_dict()
    geometry_dict["default"] = DEFAULT_PALOUSE_GEOMETRY

    return {
        "id": process_id,
        "summary": "Delineate field boundaries from Sentinel-2 imagery.",
        "description": DEFAULT_DESCRIPTION,
        "default_job_options": DEFAULT_JOB_OPTIONS,
        "parameters": [
            geometry_dict,
            temporal_extent.to_dict(),
            iou_threshold.to_dict(),
            confidence_threshold.to_dict(),
        ],
        "returns": {
            "description": "Vector cube with one polygon per detected field.",
            "schema": {"type": "object", "subtype": "datacube", "dimensions": [{"type": "geometry"}]},
        },
        "process_graph": cube.flat_graph(),
    }


def write_udp(output_path: Path = OUTPUT_PATH, **kwargs) -> Path:
    output_path.write_text(json.dumps(build_udp(**kwargs), indent=2), encoding="utf-8")
    return output_path



if __name__ == "__main__":
    written = write_udp()
    print(f"Wrote UDP to {written}")


# %%
