#%%

"""Process graph: Delineate-Anything field boundary detection via PyTorch on openEO.


Example
-------
    import openeo
    from openeo_udp.process_graph.delineate_pytorch import (
        build_delineate_pytorch,
        DEFAULT_JOB_OPTIONS,
    )

    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()

    cube = build_delineate_pytorch(
        connection=conn,
        geometry={
            "type": "Polygon",
            "coordinates": [[[5.0, 51.0], [5.1, 51.0], [5.1, 51.1], [5.0, 51.1], [5.0, 51.0]]],
        },
        temporal_extent=["2024-04-01", "2024-09-30"],
    )
    cube.execute_batch(
        outputfile="delineate_fields.geojson",
        out_format="GeoJSON",
        job_options=DEFAULT_JOB_OPTIONS,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import openeo



# ---------------------------------------------------------------------------
# UDF dependency archives
# ---------------------------------------------------------------------------
# ONNX runtime dependencies archive (Python 3.11).
DEFAULT_ONNX_DEPS_ARCHIVE_URL = (
    "https://s3.waw3-1.cloudferro.com/"
    "project_dependencies/onnx_deps_python311.zip#onnx_deps"
)

# Uploaded model archive mounted as ``onnx_models/``.
# Keep this filename aligned with what you published in project_dependencies.
DEFAULT_MODEL_ARCHIVE_URL = (
    "https://s3.waw3-1.cloudferro.com/"
    "project_dependencies/DelineateAnythingv2.zip#onnx_models"
)

print("Using ONNX deps archive:", DEFAULT_ONNX_DEPS_ARCHIVE_URL)
print("Using model archive:", DEFAULT_MODEL_ARCHIVE_URL)

DEFAULT_JOB_OPTIONS: dict = {
    "udf-dependency-archives": [
        DEFAULT_ONNX_DEPS_ARCHIVE_URL,
        DEFAULT_MODEL_ARCHIVE_URL,
    ]
}

# Model reference path inside the mounted archive fragment.
DEFAULT_WEIGHTS_URL = "onnx_models/DelineateAnythingv2.onnx"

# The model processes 512x512 tiles.  inner=384 + overlap=64 each side → the
# UDF receives 512x512 tiles (384 + 2*64).  Overlap lets instance masks that
# straddle tile boundaries be reconciled downstream.
CHUNK_INNER_PX = 384
CHUNK_OVERLAP_PX = 64

# Detection / post-processing defaults (validated locally on BAP_input.nc).
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.7  # ultralytics default, matches the reference pipeline
MORPHOLOGY = False

UDF_PATH = Path(__file__).resolve().parents[1] / "udf" / "delineate_onnx.py"
SORT_UDF_PATH = UDF_PATH.with_name("sort_fields.py")

# S2 bands for RGB (true colour: B04, B03, B02).
S2_RGB_BANDS = ["B04", "B03", "B02"]




#%%
# ---------------------------------------------------------------------------
# BAP composite loader (same as delineate_onnx.py)
# ---------------------------------------------------------------------------

def _load_bap_composite(
    connection: openeo.Connection,
    geometry,
    temporal_extent,
    max_cloud_cover: int = 75,
) -> openeo.DataCube:
    """Load a BAP composite, reduce time (first), and scale to [0, 1]."""
    composite = connection.datacube_from_process(
        process_id="bap_composite",
        namespace=(
            "https://raw.githubusercontent.com/ESA-APEx/apex_algorithms/"
            "refs/heads/main/algorithm_catalog/vito/bap_composite/openeo_udp/"
            "bap_composite.json"
        ),
        geometry=geometry,
        temporal_extent=temporal_extent,
        bands=S2_RGB_BANDS,
        max_cloud_cover=max_cloud_cover,
    )

    # BAP already picks one best pixel per (x, y); if multiple timesteps
    # survive, pick the first rather than averaging (mean would blur RGB).
    composite = composite.reduce_dimension(dimension="t", reducer="first")

    # Scale S2 BOA reflectance to [0, 1].
    composite = composite.linear_scale_range(0, 3000, 0, 1)
    return composite


def build_bap_only(
    connection: openeo.Connection,
    geometry,
    temporal_extent,
    max_cloud_cover: int = 75,
) -> openeo.DataCube:
    """Return just the BAP RGB composite scaled to [0, 1] — useful for QA."""
    return _load_bap_composite(
        connection, geometry, temporal_extent, max_cloud_cover
    )


# ---------------------------------------------------------------------------
# Main process graph
# ---------------------------------------------------------------------------

def build_delineate_onnx(
    connection: openeo.Connection,
    geometry,
    temporal_extent=None,
    bap_cube: Optional[openeo.DataCube] = None,
    udf_path: Optional[str] = None,
    weights_url: str = DEFAULT_WEIGHTS_URL,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    morphology: bool = MORPHOLOGY,
    max_cloud_cover: int = 75,
    processing_options=None,
) -> openeo.DataCube:
    """Build the Delineate-Anything PyTorch inference process graph.

    Parameters
    ----------
    connection : authenticated openeo.Connection
    geometry : GeoJSON geometry (Polygon) for the area of interest
    temporal_extent : [start, end] ISO date strings
    bap_cube : optional pre-built BAP composite (skip the BAP UDP call)
    udf_path : optional override path for the UDF source file
    weights_url : HTTPS URL to a Delineate-Anything YOLO .pt checkpoint
    confidence_threshold : YOLO detection confidence
    iou_threshold : NMS IoU threshold
    morphology : apply 3x3 open+close on the per-instance masks
    processing_options : optional UDF context dict/Parameter that overrides
                         any of the above at runtime

    Returns
    -------
    Vector cube, one polygon per field: ``band0`` (chunk-coded ID) and
    ``field_id`` (1..N, north->south, west->east), chunk-seam splits merged.
    """
    if bap_cube is not None:
        composite = bap_cube
    else:
        composite = _load_bap_composite(
            connection, geometry, temporal_extent, max_cloud_cover
        )

    udf_src_path = Path(udf_path) if udf_path else UDF_PATH
    udf_code = udf_src_path.read_text(encoding="utf-8")

    default_context = {
        "weights_url": weights_url,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "morphology": morphology,
        "chunk_core_px": CHUNK_INNER_PX,
    }
    context = processing_options if processing_options is not None else default_context

    detected = composite.apply_neighborhood(
        process=openeo.UDF(udf_code, runtime="Python", context=context),
        size=[
            {"dimension": "x", "value": CHUNK_INNER_PX, "unit": "px"},
            {"dimension": "y", "value": CHUNK_INNER_PX, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": CHUNK_OVERLAP_PX, "unit": "px"},
            {"dimension": "y", "value": CHUNK_OVERLAP_PX, "unit": "px"},
        ],
    )

    
    # Background (0) -> nodata so only fields are polygonised.
    fields = detected.mask(detected == 0).raster_to_vector()
    # Second UDF: merge polygons split by chunk seams, add field_id 1..N.
    sort_udf = openeo.UDF(SORT_UDF_PATH.read_text(encoding="utf-8"), runtime="Python")
    return fields.apply_dimension(process=sort_udf, dimension="geometry")



def main() -> None:
    backend = "https://openeo.dataspace.copernicus.eu"
    weights_url = DEFAULT_WEIGHTS_URL
    result_dir = Path("delineate_fields_result")

    conn = openeo.connect(backend)
    conn.authenticate_oidc()

    # Palouse (SE Washington, USA), ~10 km x 10 km around rolling croplands.
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-117.0635, 46.7150],
            [-116.9325, 46.7150],
            [-116.9325, 46.8050],
            [-117.0635, 46.8050],
            [-117.0635, 46.7150],
        ]],
    }
    

    cube = build_delineate_onnx(
        connection=conn,
        geometry=geometry,
        temporal_extent=["2024-05-01", "2024-08-31"],
        weights_url=weights_url,
    )

    job = cube.create_job(
        out_format="GeoJSON",
        title="Delineate-Anything demo (Palouse 10x10 km)",
        job_options=DEFAULT_JOB_OPTIONS,
    )
    job.start_and_wait()
    job.get_results().download_files(result_dir)
    print(f"Job {job.job_id} results in {result_dir}")


#%%
def plot_vectorcube_geojson(result_dir: Path) -> None:
    """Plot the downloaded vectorcube GeoJSON with field_id labels."""
    import geopandas as gpd
    import matplotlib.pyplot as plt

    vector_path = result_dir / "vectorcube.geojson"
    if not vector_path.exists():
        print(f"No vectorcube result found at: {vector_path}")
        return

    fields = gpd.read_file(vector_path)
    if fields.empty:
        print(f"Vector cube is empty: {vector_path}")
        return

    column = "field_id" if "field_id" in fields.columns else "band0"
    ax = fields.plot(
        column=column,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.2,
        figsize=(10, 10),
    )
    for fid, pt in zip(fields[column], fields.representative_point()):
        ax.annotate(str(fid), (pt.x, pt.y), ha="center", va="center", fontsize=4)
    ax.set_title(f"{len(fields)} fields ({column})")
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    main()
    plot_vectorcube_geojson(Path("delineate_fields_result"))

# %%
