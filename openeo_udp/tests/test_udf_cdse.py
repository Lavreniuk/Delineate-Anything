#%%

"""Process graph: Delineate-Anything field boundary detection via PyTorch on openEO.

Mirrors the TESSERA-v2 openEO example (torch-in-UDF pattern):

    BAP composite (RGB, [0,1])
        │
        └─► apply_neighborhood(size=512x512)
              └─► UDF: torch + ultralytics YOLO.predict(retina_masks=True)
                    └─► instance label map (uint16 IDs in a float32 band)

Compared to ``delineate_onnx.py``:
  * Uses PyTorch + Ultralytics inside the UDF (via ``udf-dependency-archives``)
    instead of raw ONNX runtime + hand-rolled YOLO decoding.
  * The UDF returns a clean instance label map, not a soft heatmap, so no
    fragile ``mask > 0.2`` step is needed downstream.

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
        outputfile="delineate_fields.tif",
        out_format="GTiff",
        job_options=DEFAULT_JOB_OPTIONS,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import openeo
import rasterio

try:
    from rasterio.merge import merge as merge_rasters
except ImportError:  # pragma: no cover
    merge_rasters = None



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
IOU_THRESHOLD = 0.3
MORPHOLOGY = False

from pathlib import Path
UDF_PATH = Path('C:\\Git_projects\\Delineate-Anything\\openeo_udp\\udf\\delineate_onnx.py')

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
    openeo.DataCube with 1 band (``instances``): float32 integer instance IDs
    (0 = background).  Downstream ops can polygonize, filter by area, or
    re-label across tile boundaries.
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
    return detected


def _load_instance_raster(path: Path | str) -> np.ndarray:
    """Read a GeoTIFF result into a 2D label map."""
    with rasterio.open(path) as src:
        arr = src.read()

    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.int32)

    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
    arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr.astype(np.int32, copy=False)


def _recombine_chunked_instances(label_map: np.ndarray) -> np.ndarray:
    """Merge chunk-local instance IDs into a single connected-component map.

    The UDF is run on overlapping tiles, so the same field can appear with
    different local IDs on adjacent chunks.  We collapse the foreground mask into
    connected components, which gives one consistent instance ID across chunk
    seams when the pixels are spatially connected.
    """
    fg = label_map > 0
    if not fg.any():
        return np.zeros_like(label_map, dtype=np.int32)

    h, w = fg.shape
    visited = np.zeros_like(fg, dtype=bool)
    merged = np.zeros_like(label_map, dtype=np.int32)
    current_id = 1

    for y in range(h):
        for x in range(w):
            if not fg[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                merged[cy, cx] = current_id
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if fg[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            current_id += 1

    return merged


def _download_result_tiles(job: openeo.BatchJob, output_dir: Path | str) -> list[Path]:
    """Download all GTiff outputs from a batch job."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        job.download_results(str(out_dir))
    except TypeError:
        job.download_results(output_dir=str(out_dir))

    tif_files = sorted(out_dir.glob("*.tif")) + sorted(out_dir.glob("*.tiff"))
    tif_files = sorted({p.resolve() for p in tif_files})
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF outputs found in {out_dir}")
    return tif_files


def _mosaic_result_tiles(tif_files: list[Path]) -> np.ndarray:
    """Mosaic multiple result tiles into one array if the job produced several files."""
    if len(tif_files) == 1:
        return _load_instance_raster(tif_files[0])

    datasets = [rasterio.open(p) for p in tif_files]
    try:
        if merge_rasters is None:
            raise RuntimeError("rasterio.merge is unavailable")
        mosaic, _ = merge_rasters(datasets, nodata=0)
    finally:
        for ds in datasets:
            ds.close()

    if mosaic.ndim == 3:
        mosaic = mosaic[0] if mosaic.shape[0] == 1 else mosaic.transpose(1, 2, 0)
    arr = np.asarray(mosaic)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32, copy=False)


def _visualize_instance_map(label_map: np.ndarray, title: str, save_path: Path | str) -> None:
    """Create a simple overlay plot of the final merged instances."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    masked = np.ma.masked_where(label_map == 0, label_map)
    ax.imshow(masked, cmap="tab20", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _download_and_visualize_batch_result(job: openeo.BatchJob, output_dir: Path | str) -> np.ndarray:
    """Download a batch result, recombine chunk-local labels, and save a preview plot."""
    output_dir = Path(output_dir)
    tile_paths = _download_result_tiles(job, output_dir)
    raw_labels = _mosaic_result_tiles(tile_paths)
    merged_labels = _recombine_chunked_instances(raw_labels)

    preview_path = output_dir / "merged_instances_preview.png"
    _visualize_instance_map(merged_labels, "Merged chunked instance labels", preview_path)

    print(f"Downloaded result files: {[p.name for p in tile_paths]}")
    print(f"Raw label stats: min={raw_labels.min() if raw_labels.size else 0}, max={raw_labels.max() if raw_labels.size else 0}")
    print(f"Merged label stats: min={merged_labels.min() if merged_labels.size else 0}, max={merged_labels.max() if merged_labels.size else 0}")
    print(f"Saved merged preview: {preview_path}")
    return merged_labels


def main() -> None:
    backend = "https://openeo.dataspace.copernicus.eu"
    output_path = "delineate_fields.tif"
    weights_url = DEFAULT_WEIGHTS_URL
    result_dir = Path("delineate_fields_result")

    conn = openeo.connect(backend)
    conn.authenticate_oidc()

    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [4.997886465978867, 51.000003053347534],
            [5.0500127304842835, 50.99909971909893],
            [5.052258312621218, 51.04996441050298],
            [5.000075043015026, 51.05086937572316],
            [4.997886465978867, 51.000003053347534],
        ]],
    }

    cube = build_delineate_onnx(
        connection=conn,
        geometry=geometry,
        temporal_extent=["2024-05-01", "2024-08-31"],
        weights_url=weights_url,
    )

    job = cube.execute_batch(
        outputfile=output_path,
        out_format="GTiff",
        title="Delineate-Anything demo (PyTorch UDF)",
        job_options=DEFAULT_JOB_OPTIONS,
    )
    print(f"Submitted job {job.job_id}")
    print("Waiting for job completion...")
    job.get_results()

    merged_labels = _download_and_visualize_batch_result(job, result_dir)
    print(f"Downloaded merged instance map with {int(merged_labels.max())} objects.")


if __name__ == "__main__":
    main()

#%%

