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

# Post-processing operates on a larger extent (multiple inference tiles).
# 2048px inner + 64px overlap = 2176px chunks → connected components merge
# fields that span across inference tile boundaries.
POSTPROC_INNER_PX = 2048
POSTPROC_OVERLAP_PX = 64

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.005

# Post-processing defaults
MASK_THRESHOLD = 0.5
MIN_AREA_PX = 50
MIN_HOLE_AREA_PX = 25

UDF_PATH = Path(__file__).resolve().parent.parent / "udf" / "delineate_inference.py"
POSTPROC_UDF_PATH = Path(__file__).resolve().parent.parent / "udf" / "delineate_postprocess.py"

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
    """Load a BAP composite, reduce time, and scale to [0, 1].

    Steps:
      1. Call the APEx bap_composite UDP for RGB bands.
      2. Reduce the temporal dimension via mean (in case multiple dates remain).
      3. Scale S2 BOA reflectance from [0, 3000] to [0, 1].

    See: https://algorithm-catalogue.apex.esa.int/apps/bap_composite
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

    # Temporal mean (BAP should already be single-date, but just in case)
    composite = composite.reduce_dimension(dimension="t", reducer="mean")

    # Scale S2 BOA reflectance to [0, 1]
    composite = composite.linear_scale_range(0, 3000, 0, 1)

    return composite


def build_bap_only(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    max_cloud_cover: int = 75,
) -> openeo.DataCube:
    """Build just the BAP composite scaled to [0, 1].

    Useful for inspecting / debugging the input to the inference UDF.
    """
    return _load_bap_composite(
        connection, spatial_extent, temporal_extent, max_cloud_cover
    )


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


def build_delineate_full(
    connection: openeo.Connection,
    spatial_extent: dict,
    temporal_extent: list[str],
    bap_cube: Optional[openeo.DataCube] = None,
    udf_path: Optional[str] = None,
    postproc_udf_path: Optional[str] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    mask_threshold: float = MASK_THRESHOLD,
    min_area_px: int = MIN_AREA_PX,
    min_hole_area_px: int = MIN_HOLE_AREA_PX,
) -> openeo.DataCube:
    """Build the full pipeline: BAP → inference → post-processing.

    The post-processing UDF runs on a larger spatial extent (2048×2048 inner)
    so that connected-component labeling merges fields that span across
    inference tile boundaries.

    Parameters
    ----------
    connection : authenticated openeo.Connection
    spatial_extent : dict with west/south/east/north[/crs]
    temporal_extent : [start, end] ISO date strings
    bap_cube : optional pre-built BAP composite datacube
    udf_path : override for inference UDF
    postproc_udf_path : override for post-processing UDF
    confidence_threshold : YOLO detection confidence threshold
    mask_threshold : binarization threshold for instance segmentation
    min_area_px : minimum field area in pixels
    min_hole_area_px : minimum hole area to keep

    Returns
    -------
    openeo.DataCube with 1 output band (instances)
    """
    # Step 1: build the composite (already scaled to [0, 1])
    if bap_cube is not None:
        composite = bap_cube
    else:
        composite = _load_bap_composite(connection, spatial_extent, temporal_extent)

    # Step 2: inference
    detected = build_delineate_onnx(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bap_cube=composite,
        udf_path=udf_path,
        confidence_threshold=confidence_threshold,
    )

    # Step 2: post-processing on larger tiles
    postproc_src = Path(postproc_udf_path) if postproc_udf_path else POSTPROC_UDF_PATH
    postproc_code = postproc_src.read_text(encoding="utf-8")

    instances = detected.apply_neighborhood(
        process=openeo.UDF(
            postproc_code,
            runtime="Python",
            context={
                "mask_threshold": mask_threshold,
                "min_area_px": min_area_px,
                "min_hole_area_px": min_hole_area_px,
            },
        ),
        size=[
            {"dimension": "x", "value": POSTPROC_INNER_PX, "unit": "px"},
            {"dimension": "y", "value": POSTPROC_INNER_PX, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": POSTPROC_OVERLAP_PX, "unit": "px"},
            {"dimension": "y", "value": POSTPROC_OVERLAP_PX, "unit": "px"},
        ],
    )

    return instances
