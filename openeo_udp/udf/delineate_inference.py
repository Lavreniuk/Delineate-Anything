"""OpenEO UDF: Delineate-Anything field boundary detection (ONNX inference).

This UDF receives a Best-Available Pixel (BAP) RGB composite datacube and
runs YOLO-seg ONNX inference on 512x512 tiles to produce per-pixel field
boundary confidence scores.

The model is a YOLOv11 instance-segmentation network trained on diverse
satellite imagery (Sentinel-2, Planet, Maxar, Google) for agricultural
field boundary delineation.

Post-processing (NMS, mask assembly, polygon merging) is NOT performed
in this UDF — only raw model outputs are returned.

Invocation
----------
Call via ``apply_neighborhood`` on chunks of shape (bands=3, y=512, x=512).
The BAP composite must be RGB (3 bands). Pixel values are expected to be
pre-scaled to [0, 1] via ``linear_scale_range(0, 3000, 0, 1)`` in the
process graph.

Dependency archives (job options ``udf-dependency-archives``)::

    onnx_deps/              -> onnxruntime wheel
    DelineateAnything/      -> DelineateAnything.onnx

Output bands (2):
    - detection   (float32, raw detection confidence per pixel)
    - mask        (float32, raw instance mask probability per pixel)

Context overrides::

    {
        "confidence_threshold": 0.005,  # YOLO detection confidence threshold
        "model_name": "DelineateAnything.onnx",
    }
"""

import functools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from openeo.metadata import CollectionMetadata

# ---------------------------------------------------------------------------
# Make UDF dependency archives importable.
# ---------------------------------------------------------------------------
sys.path.append("onnx_deps")
import onnxruntime as ort  # noqa: E402

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================
NUM_THREADS = 2
DEFAULT_MODEL_NAME = "DelineateAnything.onnx"
MODEL_DIR = "DelineateAnything"
DEFAULT_CONFIDENCE_THRESHOLD = 0.005

# Model input size (must match export --imgsz)
MODEL_INPUT_SIZE = 512


# ===========================================================================
# ONNX session loader (cached per executor)
# ===========================================================================

def _ort_session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = NUM_THREADS
    so.inter_op_num_threads = NUM_THREADS
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    return so


@functools.lru_cache(maxsize=1)
def _load_session(model_name: str = DEFAULT_MODEL_NAME) -> ort.InferenceSession:
    """Load the ONNX model from the dependency archive."""
    model_root = Path(MODEL_DIR)
    model_path = model_root / model_name

    if not model_path.exists():
        # Search recursively
        by_name = list(model_root.rglob(model_name))
        if len(by_name) == 1:
            model_path = by_name[0]
        elif len(by_name) > 1:
            raise FileNotFoundError(
                f"Multiple ONNX model matches for '{model_name}' under {model_root}."
            )
        else:
            all_onnx = list(model_root.rglob("*.onnx"))
            if len(all_onnx) == 1:
                model_path = all_onnx[0]
                logger.info("Auto-selected ONNX model: %s", model_path)
            else:
                raise FileNotFoundError(
                    f"ONNX model not found under {model_root}. Found: {all_onnx}"
                )

    logger.info("Loading ONNX model: %s", model_path)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=_ort_session_options(),
        providers=["CPUExecutionProvider"],
    )

    os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUM_THREADS))
    return session


# ===========================================================================
# Normalisation
# ===========================================================================

def _normalize_to_0_1(image: np.ndarray) -> np.ndarray:
    """Ensure image is in [0, 1] float32.

    The process graph applies linear_scale_range(0, 3000, 0, 1) BEFORE
    calling this UDF, so input should already be in [0, 1].  This function
    only handles NaN → 0 and clips stray values.
    """
    image = image.astype(np.float32)

    # Replace NaN with 0 so they don't propagate through inference.
    nan_count = int(np.isnan(image).sum())
    if nan_count > 0:
        logger.info("Replacing %d NaN pixels with 0 (%.1f%% of total)",
                    nan_count, 100.0 * nan_count / image.size)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    return np.clip(image, 0.0, 1.0)


# ===========================================================================
# YOLO post-processing (minimal — confidence map only)
# ===========================================================================

def _extract_confidence_map(
    outputs: list[np.ndarray],
    input_h: int,
    input_w: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a per-pixel confidence map from YOLO-seg raw outputs.

    YOLO-seg typically outputs:
      - output0: (1, num_classes+4+mask_dim, num_detections) — detection tensor
      - output1: (1, mask_dim, mask_h, mask_w) — mask prototypes

    We combine detection confidences with mask prototypes to produce a
    per-pixel field boundary probability map.

    Returns (detection_map, mask_map) each of shape (H, W) in float32.
    """
    if len(outputs) < 2:
        # Fallback: if only one output, return zeros
        logger.warning("Expected 2 outputs (detections + masks), got %d", len(outputs))
        return (
            np.zeros((input_h, input_w), dtype=np.float32),
            np.zeros((input_h, input_w), dtype=np.float32),
        )

    # output0: (1, 4+nc+mask_dim, num_detections) — transposed format
    det_output = outputs[0]  # (1, C, N)
    mask_protos = outputs[1]  # (1, mask_dim, mh, mw)

    if det_output.ndim == 3:
        det_output = det_output[0]  # (C, N)
    if mask_protos.ndim == 4:
        mask_protos = mask_protos[0]  # (mask_dim, mh, mw)

    mask_dim = mask_protos.shape[0]
    n_det_channels = det_output.shape[0]
    n_detections = det_output.shape[1]

    # Channels: [x, y, w, h, class_conf..., mask_coeffs...]
    # For single-class (field boundary): channels = 4 + 1 + mask_dim
    n_classes = n_det_channels - 4 - mask_dim
    logger.info("YOLO decode: det_channels=%d, mask_dim=%d, n_classes=%d, n_detections=%d",
                n_det_channels, mask_dim, n_classes, n_detections)
    logger.info("Mask protos shape: (%d, %d, %d)", mask_dim, mask_protos.shape[1], mask_protos.shape[2])

    # Extract components
    # boxes = det_output[:4, :]  # (4, N) — not needed for pixel map
    class_scores = det_output[4:4 + n_classes, :]  # (nc, N)
    mask_coeffs = det_output[4 + n_classes:, :]  # (mask_dim, N)

    # Max class confidence per detection
    confidences = class_scores.max(axis=0)  # (N,)

    # Filter by confidence
    keep = confidences > confidence_threshold
    if not np.any(keep):
        return (
            np.zeros((input_h, input_w), dtype=np.float32),
            np.zeros((input_h, input_w), dtype=np.float32),
        )

    kept_conf = confidences[keep]  # (K,)
    kept_mask_coeffs = mask_coeffs[:, keep]  # (mask_dim, K)
    logger.info("Detections kept: %d / %d (threshold=%.4f, max_conf=%.4f)",
                int(keep.sum()), n_detections, confidence_threshold, float(confidences.max()))

    # Compute instance masks: sigmoid(coeffs^T @ protos)
    # mask_protos: (mask_dim, mh, mw)
    mh, mw = mask_protos.shape[1], mask_protos.shape[2]
    protos_flat = mask_protos.reshape(mask_dim, -1)  # (mask_dim, mh*mw)
    raw_masks = kept_mask_coeffs.T @ protos_flat  # (K, mh*mw)
    raw_masks = raw_masks.reshape(-1, mh, mw)  # (K, mh, mw)

    # Sigmoid activation
    instance_masks = 1.0 / (1.0 + np.exp(-raw_masks))  # (K, mh, mw)

    # Weighted combination: each pixel gets max(conf * mask) across detections
    # Weight masks by confidence
    weighted = instance_masks * kept_conf[:, None, None]  # (K, mh, mw)
    combined_mask = weighted.max(axis=0)  # (mh, mw)
    detection_map = np.max(instance_masks * (kept_conf[:, None, None] > 0), axis=0)

    # Resize to input resolution
    from scipy.ndimage import zoom

    scale_y = input_h / mh
    scale_x = input_w / mw
    if scale_y != 1.0 or scale_x != 1.0:
        combined_mask = zoom(combined_mask, (scale_y, scale_x), order=1)
        detection_map = zoom(detection_map, (scale_y, scale_x), order=1)

    # Ensure correct shape (zoom can produce off-by-one)
    combined_mask = combined_mask[:input_h, :input_w]
    detection_map = detection_map[:input_h, :input_w]

    return (
        detection_map.astype(np.float32),
        combined_mask.astype(np.float32),
    )


# ===========================================================================
# Inference
# ===========================================================================

def _run_inference(
    session: ort.InferenceSession,
    image_rgb_hwc: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONNX inference on a single 512x512 RGB tile.

    Parameters
    ----------
    image_rgb_hwc : (H, W, 3) float32 in [0, 1]

    Returns
    -------
    (detection_map, mask_map) each (H, W) float32
    """
    h, w = image_rgb_hwc.shape[:2]

    # YOLO expects (1, 3, H, W) in [0, 1]
    image_chw = np.transpose(image_rgb_hwc, (2, 0, 1))  # (3, H, W)
    batch = image_chw[np.newaxis, ...].astype(np.float32)  # (1, 3, H, W)

    input_name = session.get_inputs()[0].name
    logger.info("Running ONNX inference: input shape=%s, min=%.4f, max=%.4f",
                batch.shape, float(batch.min()), float(batch.max()))
    outputs = session.run(None, {input_name: batch})
    logger.info("ONNX raw outputs: %d tensors, shapes=%s",
                len(outputs), [o.shape for o in outputs])
    for i, o in enumerate(outputs):
        logger.info("  output[%d]: min=%.6f, max=%.6f, mean=%.6f",
                    i, float(o.min()), float(o.max()), float(o.mean()))

    detection_map, mask_map = _extract_confidence_map(
        outputs, h, w, confidence_threshold
    )
    return detection_map, mask_map


# ===========================================================================
# OpenEO UDF entry points
# ===========================================================================

def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Declare the 5-band output schema (RGB mosaic + model outputs)."""
    return metadata.rename_labels(
        dimension="bands",
        target=["red", "green", "blue", "detection", "mask"],
    )


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """Main UDF entry point: normalise BAP RGB -> ONNX inference -> 5-band output.

    Input cube dims: (bands=3, y, x) — RGB BAP composite.
    Output cube dims: (bands=5, y, x) — RGB composite + detection + mask.
    """
    confidence_threshold = float(
        context.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    )
    model_name = context.get("model_name", DEFAULT_MODEL_NAME)

    logger.info("=== Delineate UDF START ===")
    logger.info("Input cube dims: %s, shape: %s, dtype: %s", list(cube.dims), cube.shape, cube.dtype)
    logger.info("Context: %s", context)

    dims = list(cube.dims)

    # Handle optional time dimension: reduce to single image via median
    t_dim = next((d for d in dims if d in ("t", "time")), None)
    if t_dim is not None:
        logger.info("Time dimension '%s' found with %d steps — reducing via mean", t_dim, cube.sizes[t_dim])
        if cube.sizes[t_dim] == 1:
            cube = cube.squeeze(t_dim)
        else:
            cube = cube.mean(dim=t_dim)
        dims = list(cube.dims)
        logger.info("After time reduction: dims=%s, shape=%s", dims, cube.shape)

    # Identify band and spatial dimensions
    b_dim = next((d for d in dims if d in ("bands", "band", "spectral")), None)
    if b_dim is None:
        raise ValueError(f"Cannot find band dimension in {dims}")

    spatial_dims = [d for d in dims if d != b_dim]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {spatial_dims}")
    y_dim, x_dim = spatial_dims

    # Transpose to (bands, y, x) and get numpy
    data = cube.transpose(b_dim, y_dim, x_dim).values.astype(np.float32)  # (3, H, W)
    n_bands, h, w = data.shape
    logger.info("Data shape: (%d bands, %d H, %d W)", n_bands, h, w)
    logger.info("Raw data stats: min=%.4f, max=%.4f, mean=%.4f, nan_count=%d",
                float(np.nanmin(data)), float(np.nanmax(data)), float(np.nanmean(data)),
                int(np.isnan(data).sum()))

    if n_bands < 3:
        raise ValueError(f"Expected at least 3 bands (RGB), got {n_bands}.")

    # Take only first 3 bands (RGB)
    rgb = data[:3, :, :]  # (3, H, W)
    image_hwc = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

    # Log per-band stats before normalisation
    for i, band_name in enumerate(["Red", "Green", "Blue"]):
        band = image_hwc[:, :, i]
        logger.info("%s band: min=%.4f, max=%.4f, mean=%.4f, zeros_pct=%.1f%%",
                    band_name, float(np.nanmin(band)), float(np.nanmax(band)),
                    float(np.nanmean(band)), 100.0 * (band == 0).sum() / band.size)

    # Normalise to [0, 1]
    image_hwc = _normalize_to_0_1(image_hwc)
    logger.info("After normalisation: min=%.4f, max=%.4f, mean=%.4f",
                float(image_hwc.min()), float(image_hwc.max()), float(image_hwc.mean()))

    # Check if tile is all-zero / nodata
    if image_hwc.max() == 0:
        logger.warning("Tile is all zeros — skipping inference, returning empty output")
        stacked = np.zeros((5, h, w), dtype=np.float32)
        coords: dict = {}
        if y_dim in cube.coords:
            coords[y_dim] = cube.coords[y_dim]
        if x_dim in cube.coords:
            coords[x_dim] = cube.coords[x_dim]
        return xr.DataArray(stacked, dims=("bands", y_dim, x_dim), coords=coords)

    # Run inference
    logger.info("Loading ONNX session (model=%s)...", model_name)
    session = _load_session(model_name)

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()
    logger.info("ONNX model input: name='%s', shape=%s, type=%s",
                input_info.name, input_info.shape, input_info.type)
    logger.info("ONNX model outputs: %s",
                [(o.name, o.shape, o.type) for o in output_info])

    detection_map, mask_map = _run_inference(session, image_hwc, confidence_threshold)
    logger.info("Detection map: min=%.4f, max=%.4f, mean=%.4f, nonzero_pct=%.2f%%",
                float(detection_map.min()), float(detection_map.max()),
                float(detection_map.mean()), 100.0 * (detection_map > 0).sum() / detection_map.size)
    logger.info("Mask map: min=%.4f, max=%.4f, mean=%.4f, nonzero_pct=%.2f%%",
                float(mask_map.min()), float(mask_map.max()),
                float(mask_map.mean()), 100.0 * (mask_map > 0).sum() / mask_map.size)

    # Stack output: RGB mosaic (already [0,1]) + detection + mask
    stacked = np.stack([
        image_hwc[:, :, 0],  # red
        image_hwc[:, :, 1],  # green
        image_hwc[:, :, 2],  # blue
        detection_map,
        mask_map,
    ], axis=0)  # (5, H, W)

    coords = {}
    if y_dim in cube.coords:
        coords[y_dim] = cube.coords[y_dim]
    if x_dim in cube.coords:
        coords[x_dim] = cube.coords[x_dim]

    logger.info("=== Delineate UDF END === output shape: %s", stacked.shape)
    return xr.DataArray(
        stacked,
        dims=("bands", y_dim, x_dim),
        coords=coords,
    )
