#%%
"""OpenEO UDF: Delineate-Anything field boundary detection via ONNX Runtime.

The model artefact (``DelineateAnything-S.onnx``) and the ONNX runtime are
provided as ``udf-dependency-archives`` and mounted at:

    onnx_deps/      -> onnxruntime wheel(s)
    onnx_models/    -> DelineateAnything-S.onnx (+ any siblings)

This UDF does NOT download weights at runtime and has no torch / ultralytics
dependency.  Inference and YOLO-seg post-processing (sigmoid → mask assembly
via mask prototypes, class-agnostic NMS, mask binarisation) are implemented
directly with numpy + onnxruntime.

Invocation
----------
Call via ``apply_neighborhood`` on chunks of shape ``(bands=3, y=512, x=512)``
with RGB reflectance scaled to ``[0, 1]``.

Chunked instance IDs
--------------------
The output band ``instances`` is a per-chunk label map: ``0`` is background
and ``1..N`` are the detections *inside this chunk only*. Instance IDs are
NOT globally unique across chunks — merging / re-labelling across tile
boundaries must be done downstream (e.g. by polygonising and re-numbering).

Context overrides::

    {
        "weights_path": "onnx_models/DelineateAnything-S.onnx",
        "confidence_threshold": 0.005,
        "iou_threshold": 0.3,
        "morphology": false,
    }
"""

import functools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from openeo.udf import XarrayDataCube
from openeo.metadata import CollectionMetadata


# ---------------------------------------------------------------------------
# Make mounted UDF dependency archives importable.
# ---------------------------------------------------------------------------
_ONNX_DEPS_DIR = os.environ.get("DELINEATE_ONNX_DEPS_DIR", "onnx_deps")
if os.path.isdir(_ONNX_DEPS_DIR) and _ONNX_DEPS_DIR not in sys.path:
    sys.path.insert(0, _ONNX_DEPS_DIR)

import onnxruntime as ort  # noqa: E402

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================
DEFAULT_ONNX_MODEL_NAME = "DelineateAnything-S.onnx"
DEFAULT_WEIGHTS_PATH = f"onnx_models/{DEFAULT_ONNX_MODEL_NAME}"

DEFAULT_CONFIDENCE_THRESHOLD = 0.005
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MORPHOLOGY = False
MODEL_INPUT_SIZE = 512

MAX_INSTANCES_PER_TILE = 65534
# Ultralytics predict() defaults
MAX_DET = 300      # cap on kept detections after NMS
MAX_NMS = 30_000   # cap on candidates entering NMS
NUM_THREADS = 2


# ===========================================================================
# Model artefact resolution
# ===========================================================================

def _resolve_model_path(weights_ref: str) -> Path:
    """Locate the ONNX model file inside the mounted dependency archives."""
    direct = Path(weights_ref)
    if direct.exists():
        return direct.resolve()

    roots = [
        Path(os.environ.get("DELINEATE_ONNX_MODELS_DIR", "onnx_models")),
        Path("onnx_models"),
        Path("DelineateAnything"),
    ]

    for root in roots:
        candidate = root / weights_ref
        if candidate.exists():
            return candidate.resolve()

    target_name = Path(weights_ref).name or DEFAULT_ONNX_MODEL_NAME
    for root in roots:
        if not root.exists():
            continue
        matches = list(root.rglob(target_name))
        if len(matches) == 1:
            return matches[0].resolve()

    for root in roots:
        if not root.exists():
            continue
        onnx_matches = list(root.rglob("*.onnx"))
        if len(onnx_matches) == 1:
            return onnx_matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find ONNX model for '{weights_ref}'. "
        "Mount it via udf-dependency-archives "
        "(e.g. .../Deliniate-Anything-S.zip#onnx_models)."
    )


@functools.lru_cache(maxsize=2)
def _load_session(model_path_str: str) -> ort.InferenceSession:
    """Load and cache an ONNX Runtime session for a given model artefact."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = NUM_THREADS
    so.inter_op_num_threads = NUM_THREADS
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info("Loading ONNX model: %s", model_path_str)
    session = ort.InferenceSession(
        model_path_str,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUM_THREADS))
    return session


# ===========================================================================
# YOLO-seg post-processing (numpy port of Ultralytics AutoBackend logic)
# ===========================================================================



def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xy = boxes[:, :2]
    wh = boxes[:, 2:4]
    xyxy = np.empty_like(boxes[:, :4])
    xyxy[:, 0] = xy[:, 0] - wh[:, 0] / 2.0
    xyxy[:, 1] = xy[:, 1] - wh[:, 1] / 2.0
    xyxy[:, 2] = xy[:, 0] + wh[:, 0] / 2.0
    xyxy[:, 3] = xy[:, 1] + wh[:, 1] / 2.0
    return xyxy


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Class-agnostic NMS (Non-Maximum Suppression).
        A post-processing step for  object/instance detection that produces many overlapping candidate detections. Returns indices sorted by decreasing score."""
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)

        order = rest[iou <= iou_thr]

    return np.asarray(keep, dtype=np.int64)


def _morph_open_close_3x3(mask: np.ndarray) -> np.ndarray:
    """3x3 morphological open then close on a boolean mask, numpy only."""
    def _dilate(m: np.ndarray) -> np.ndarray:
        out = m.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        out[1:, 1:] |= m[:-1, :-1]
        out[1:, :-1] |= m[:-1, 1:]
        out[:-1, 1:] |= m[1:, :-1]
        out[:-1, :-1] |= m[1:, 1:]
        return out

    def _erode(m: np.ndarray) -> np.ndarray:
        return ~_dilate(~m)

    return _dilate(_erode(_dilate(_erode(mask))))


def _bilinear_upsample(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of a 2-D float mask (torch F.interpolate align_corners=False)."""
    ih, iw = mask.shape
    if (ih, iw) == (out_h, out_w):
        return mask
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * ih / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * iw / out_w - 0.5
    y = np.clip(y, 0.0, ih - 1)
    x = np.clip(x, 0.0, iw - 1)
    y0 = np.floor(y).astype(np.int32); y1 = np.clip(y0 + 1, 0, ih - 1)
    x0 = np.floor(x).astype(np.int32); x1 = np.clip(x0 + 1, 0, iw - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    Ia = mask[y0[:, None], x0[None, :]]
    Ib = mask[y0[:, None], x1[None, :]]
    Ic = mask[y1[:, None], x0[None, :]]
    Id = mask[y1[:, None], x1[None, :]]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    return top * (1.0 - wy) + bot * wy


def _predict_tile(
    session: ort.InferenceSession,
    image_hwc: np.ndarray,
    conf_thr: float,
    iou_thr: float,
    morphology: bool,
) -> np.ndarray:
    """Run one tile through the ONNX YOLO-seg model → per-chunk label map."""
    h, w = image_hwc.shape[:2]

    x = image_hwc.astype(np.float32, copy=False)
    x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 512, 512)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})

    # Ultralytics YOLO-seg ONNX exports 2 outputs:
    #   preds:  (1, 4+nc+nm, num_anchors)   e.g. (1, 37, 5376) for nc=1, nm=32
    #   protos: (1, nm, mh, mw)             e.g. (1, 32, 128, 128)
    preds, protos = None, None
    for out in outputs:
        if out.ndim == 4:
            protos = out
        elif out.ndim == 3:
            preds = out
    if preds is None or protos is None:
        raise RuntimeError(
            f"Unexpected ONNX output shapes: {[o.shape for o in outputs]}"
        )

    preds = preds[0].astype(np.float32)       # (C, A)
    protos = protos[0].astype(np.float32)     # (nm, mh, mw)
    nm, mh, mw = protos.shape
    c_total = preds.shape[0]
    nc = c_total - 4 - nm
    if nc <= 0:
        raise RuntimeError(f"Cannot infer class count from preds shape {preds.shape}")

    preds = preds.T                            # (A, C)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:4 + nc]
    mask_coefs = preds[:, 4 + nc:]             # (A, nm)

    scores = class_scores.max(axis=1)          # class-agnostic
    conf_mask = scores >= conf_thr
    if not np.any(conf_mask):
        return np.zeros((h, w), dtype=np.int32)

    boxes_xywh = boxes_xywh[conf_mask]
    scores = scores[conf_mask]
    mask_coefs = mask_coefs[conf_mask]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    # Match ultralytics: sort by score DESC and cap pre-NMS candidates.
    order = np.argsort(-scores)[:MAX_NMS]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]
    mask_coefs = mask_coefs[order]

    keep = _nms(boxes_xyxy, scores, iou_thr)
    if keep.size == 0:
        return np.zeros((h, w), dtype=np.int32)

    # Match ultralytics `max_det=300`.
    keep = keep[:MAX_DET]
    boxes_xyxy = boxes_xyxy[keep]
    mask_coefs = mask_coefs[keep]

    n = min(mask_coefs.shape[0], MAX_INSTANCES_PER_TILE)
    boxes_xyxy = boxes_xyxy[:n]
    mask_coefs = mask_coefs[:n]

    # Match ultralytics `process_mask_native` exactly:
    #   logits = coefs @ proto                (low res, NO sigmoid)
    #   logits = bilinear_upsample(logits)    (to model input res)
    #   masks  = crop(logits, bbox)           (zero outside bbox at full res)
    #   masks  = logits > 0                   (equivalent to sigmoid(x) > 0.5)
    # Interpolating raw logits keeps boundaries sharp; interpolating sigmoid
    # probabilities softens them and shifts the effective threshold.
    logits_lo = (mask_coefs @ protos.reshape(nm, mh * mw)).reshape(-1, mh, mw)

    label_map = np.zeros((h, w), dtype=np.int32)

    # Iterate from lowest to highest score so higher-score detections
    # overwrite overlapping lower-score ones (matches ultralytics behaviour).
    for i in range(n - 1, -1, -1):
        x1f, y1f, x2f, y2f = boxes_xyxy[i]

        # Bilinear upsample logits to model input resolution.
        logits_hi = _bilinear_upsample(logits_lo[i], MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        # Crop by the full-res bbox and threshold at 0 (== sigmoid > 0.5).
        ix1 = int(np.clip(np.floor(x1f), 0, MODEL_INPUT_SIZE - 1))
        iy1 = int(np.clip(np.floor(y1f), 0, MODEL_INPUT_SIZE - 1))
        ix2 = int(np.clip(np.ceil(x2f),  0, MODEL_INPUT_SIZE))
        iy2 = int(np.clip(np.ceil(y2f),  0, MODEL_INPUT_SIZE))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        bin_full = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=bool)
        bin_full[iy1:iy2, ix1:ix2] = logits_hi[iy1:iy2, ix1:ix2] > 0.0

        # Crop to actual output shape (tile may be smaller than 512).
        bin_mask = bin_full[:h, :w]
        if morphology and bin_mask.any():
            bin_mask = _morph_open_close_3x3(bin_mask)
        if not bin_mask.any():
            continue

        label_map[bin_mask] = i + 1

    return label_map


# ===========================================================================
# openEO UDF entry points
# ===========================================================================

def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Declare a single ``instances`` output band."""
    return metadata.rename_labels(dimension="bands", target=["instances"])


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """Main UDF entry point: RGB tile → per-chunk instance label map."""
    context = context or {}
    weights_ref = str(
        context.get("weights_path")
        or context.get("weights_url")
        or DEFAULT_WEIGHTS_PATH
    )
    conf = float(context.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    iou = float(context.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
    morphology = bool(context.get("morphology", DEFAULT_MORPHOLOGY))

    if hasattr(cube, "get_array"):
        cube = cube.get_array()

    logger.info("=== Delineate ONNX UDF START ===")
    logger.info("Input cube dims: %s, shape: %s", list(cube.dims), cube.shape)

    dims = list(cube.dims)

    t_dim = next((d for d in dims if d in ("t", "time")), None)
    if t_dim is not None:
        if cube.sizes[t_dim] == 1:
            cube = cube.squeeze(t_dim)
        else:
            cube = cube.mean(dim=t_dim)
        dims = list(cube.dims)

    b_dim = next((d for d in dims if d in ("bands", "band", "spectral")), None)
    if b_dim is None:
        raise ValueError(f"Cannot find band dimension in {dims}")
    spatial_dims = [d for d in dims if d != b_dim]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {spatial_dims}")
    y_dim, x_dim = spatial_dims

    data = cube.transpose(b_dim, y_dim, x_dim).values.astype(np.float32)
    n_bands, h, w = data.shape
    if n_bands < 3:
        raise ValueError(f"Expected ≥3 bands (RGB), got {n_bands}.")

    image_hwc = np.transpose(data[:3], (1, 2, 0))

    # openEO's apply_neighborhood centre-pads edge chunks with NaN so every
    # tile arrives at the requested (512, 512) shape. Replace with the
    # ultralytics  constant 114/255 so the model sees it as neutral
    # gray padding, then mask any detections in the padded region back out.
    nan_mask = ~np.isfinite(image_hwc).all(axis=2)  # (H, W) bool
    image_hwc = np.where(
        np.isfinite(image_hwc), image_hwc, np.float32(114.0 / 255.0)
    )

    finite_max = float(image_hwc.max()) if image_hwc.size else 0.0
    finite_min = float(image_hwc.min()) if image_hwc.size else 0.0
    if finite_max > 1.5 or finite_min < -0.1:
        raise ValueError(
            f"Input tile values out of the expected [0, 1] range "
            f"(min={finite_min:.3f}, max={finite_max:.3f}, dtype={cube.dtype}). "
            "Apply linear_scale_range(0, 3000, 0, 1) (or equivalent) upstream."
        )
    image_hwc = np.clip(image_hwc, 0.0, 1.0)

    if nan_mask.all() or image_hwc.max() == 0.0:
        logger.info("Tile is entirely padded/empty — returning empty label map")
        label_map = np.zeros((h, w), dtype=np.int32)
    else:
        pad_h = max(0, MODEL_INPUT_SIZE - h)
        pad_w = max(0, MODEL_INPUT_SIZE - w)
        if pad_h or pad_w:
            # Fallback for callers that hand us sub-512 tiles directly (e.g.
            # local tests).  Pad with 114/255 to match the openEO NaN-pad
            # convention above, not with 0.
            image_hwc = np.pad(
                image_hwc,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=114.0 / 255.0,
            )

        model_path = _resolve_model_path(weights_ref)
        session = _load_session(str(model_path))
        label_map = _predict_tile(session, image_hwc, conf, iou, morphology)

        if pad_h or pad_w:
            label_map = label_map[:h, :w]

        # Zero out any labels that fell inside the openEO-padded (NaN) region.
        if nan_mask.any():
            label_map[nan_mask] = 0

    n_fields = int(label_map.max())
    logger.info(
        "Tile done: %d fields, foreground pct=%.2f%%",
        n_fields, 100.0 * (label_map > 0).sum() / max(label_map.size, 1),
    )

    out = label_map.astype(np.float32)[np.newaxis, :, :]

    coords = {}
    if y_dim in cube.coords:
        coords[y_dim] = cube.coords[y_dim]
    if x_dim in cube.coords:
        coords[x_dim] = cube.coords[x_dim]

    logger.info("=== Delineate ONNX UDF END ===")

    result = xr.DataArray(
        out,
        dims=["bands", "y", "x"],
        coords=coords,
    )
    return XarrayDataCube(result)


# %%
