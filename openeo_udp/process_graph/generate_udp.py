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
    DEFAULT_WEIGHTS_URL,
    build_delineate_onnx,
)

DEFAULT_PROCESS_ID = "delineate_anything"
DEFAULT_BACKEND = "https://openeo.dataspace.copernicus.eu"
OUTPUT_PATH = Path(__file__).with_name(f"{DEFAULT_PROCESS_ID}_udp.json")


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
    temporal_extent = Parameter(
        name="temporal_extent",
        description="Temporal interval as [start, end] ISO-8601 date strings.",
        schema={
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
    )

    conn = openeo.connect(backend)
    cube = build_delineate_onnx(
        connection=conn,
        geometry=geometry,
        temporal_extent=temporal_extent,
        weights_url=weights_url,
    )

    return {
        "id": process_id,
        "summary": "Delineate field boundaries from Sentinel-2 imagery.",
        "description": (
            "Runs the Delineate-Anything ONNX model on a BAP RGB composite for the "
            "given geometry and temporal extent."
        ),
        "parameters": [geometry.to_dict(), temporal_extent.to_dict()],
        "process_graph": cube.flat_graph(),
    }


def write_udp(output_path: Path = OUTPUT_PATH, **kwargs) -> Path:
    output_path.write_text(json.dumps(build_udp(**kwargs), indent=2), encoding="utf-8")
    return output_path



if __name__ == "__main__":
    written = write_udp()
    print(f"Wrote UDP to {written}")


# %%
