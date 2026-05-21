"""Process graph: Delineate-Anything field boundary detection via ONNX on openEO.

Loads a Best-Available Pixel (BAP) composite (RGB), and runs
``apply_neighborhood`` with the Delineate-Anything ONNX inference UDF on
512x512 tiles.

The BAP composite is produced by the existing APEx BAP service
(https://algorithm-catalogue.apex.esa.int/apps/bap_composite) or any
other process that yields a single cloud-free RGB image.

Example
-------
    import openeo
    from openeo_udp.process_graph.delineate_onnx import (
        build_delineate_onnx,
        DEFAULT_JOB_OPTIONS,
    )

    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()

    cube = build_delineate_onnx(
        connection=conn,
        spatial_extent={"west": 5.0, "south": 51.0,
                        "east": 5.1, "north": 51.1, "crs": "EPSG:4326"},
        temporal_extent=["2024-04-01", "2024-09-30"],
    )
    cube.execute_batch(
        outputfile="delineate_fields.tif",
        out_format="GTiff",
        job_options=DEFAULT_JOB_OPTIONS,
    )
"""


from pathlib import Path
from typing import Optional

import openeo

# ---------------------------------------------------------------------------
# UDF dependency archives
# ---------------------------------------------------------------------------
# ONNX runtime wheels (Python 3.11 compatible, pre-packaged for openEO)
DEFAULT_ONNX_DEPS_ARCHIVE_URL = (
    "https://s3.waw3-1.cloudferro.com/"
    "project_dependencies/onnx_deps_python311.zip#onnx_deps"
)

# Delineate-Anything model archive: contains DelineateAnything.onnx
DEFAULT_MODEL_ARCHIVE_URL = (
    "https://s3.waw3-1.cloudferro.com/"
    "project_dependencies/DelineateAnything.zip#DelineateAnything"
)

DEFAULT_JOB_OPTIONS: dict = {
    "udf-dependency-archives": [
        DEFAULT_ONNX_DEPS_ARCHIVE_URL,
        DEFAULT_MODEL_ARCHIVE_URL,
    ],
    # YOLO-seg on 512x512 is heavier than a U-Net on 256x256
    "executor-memory": "4g",
    "executor-memoryOverhead": "4g",
    "python-memory": "disable",
    "soft-errors": 0.1,
}

# The model processes 512x512 tiles. We use inner=448 + overlap=32 on each side
# so the UDF receives exactly 512x512 (448 + 2*32 = 512).
CHUNK_INNER_PX = 448
CHUNK_OVERLAP_PX = 32

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.005

UDF_PATH = Path(__file__).resolve().parent.parent / "udf" / "delineate_inference.py"

# S2 bands for RGB (true colour: B04, B03, B02)
S2_RGB_BANDS = ["B04", "B03", "B02"]


def _spatial_extent_to_geojson(spatial_extent: dict) -> dict:
    """Convert a west/south/east/north bbox dict to a GeoJSON Polygon."""
    w = spatial_extent["west"]
    s = spatial_extent["south"]
    e = spatial_extent["east"]
    n = spatial_extent["north"]
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


def _load_bap_composite(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    max_cloud_cover: int = 75,
) -> openeo.DataCube:
    """Load a BAP composite using the APEx bap_composite UDP.

    See: https://algorithm-catalogue.apex.esa.int/apps/bap_composite

    The UDP is hosted on openeofed.dataspace.copernicus.eu with process
    ID ``bap_composite``. It takes a GeoJSON geometry, temporal extent,
    bands, and max cloud cover.
    """
    geometry = _spatial_extent_to_geojson(spatial_extent)

    composite = connection.datacube_from_process(
        process_id="bap_composite",
        namespace="https://raw.githubusercontent.com/ESA-APEx/apex_algorithms/refs/heads/main/algorithm_catalog/vito/bap_composite/openeo_udp/bap_composite.json",
        geometry=geometry,
        temporal_extent=temporal_extent,
        bands=S2_RGB_BANDS,
        max_cloud_cover=max_cloud_cover,
    )
    return composite


def build_delineate_onnx(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    bap_cube: Optional[openeo.DataCube] = None,
    udf_path: Optional[str] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> openeo.DataCube:
    """Build the Delineate-Anything ONNX inference process graph.

    Parameters
    ----------
    connection : authenticated openeo.Connection
    spatial_extent : dict with west/south/east/north[/crs]
    temporal_extent : [start, end] ISO date strings
    bap_cube : optional pre-built BAP composite datacube. If None, a simple
               cloud-masked median composite is built from S2 L2A.
    udf_path : optional override path for the UDF source file
    confidence_threshold : YOLO detection confidence threshold

    Returns
    -------
    openeo.DataCube with 2 output bands (detection, mask)
    """
    if bap_cube is not None:
        composite = bap_cube
    else:
        composite = _load_bap_composite(connection, spatial_extent, temporal_extent)

    # BAP produces one composite per month. Reduce the temporal dimension
    # to a single image (median across months) so the model gets one RGB tile.
    composite = composite.reduce_dimension(dimension="t", reducer="mean")

    udf_src_path = Path(udf_path) if udf_path else UDF_PATH
    udf_code = udf_src_path.read_text(encoding="utf-8")

    # apply_neighborhood: tile the composite into 512x512 chunks
    # inner=448, overlap=32 each side → UDF receives 512x512
    detected = composite.apply_neighborhood(
        process=openeo.UDF(
            udf_code,
            runtime="Python",
            context={
                "confidence_threshold": confidence_threshold,
            },
        ),
        size=[
            {"dimension": "x", "value": CHUNK_INNER_PX, "unit": "px"},
            {"dimension": "y", "value": CHUNK_INNER_PX, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": CHUNK_OVERLAP_PX, "unit": "px"},
            {"dimension": "y", "value": CHUNK_OVERLAP_PX, "unit": "px"},
        ],
    )
    return detected
