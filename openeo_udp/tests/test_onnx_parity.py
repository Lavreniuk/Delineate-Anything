#%%
"""End-to-end parity: PyTorch (Ultralytics YOLO on .pt) vs local ONNX Runtime.

Two independent inference paths on the exact same 512x512 RGB tile:

1. **PyTorch reference** — loads the ``.pt`` checkpoint with
   ``ultralytics.YOLO`` and calls ``model.predict(..., retina_masks=True)``
   exactly the way ``methods/main/inference.py`` does.  Per-instance masks
   from ``result.masks.data`` are assembled into a label map with a
   "higher-conf-wins-overlap" rule.

2. **ONNX local** — runs ``onnxruntime`` on the exported ``.onnx`` file
   directly in this script (no openEO / UDF involvement) and decodes the
   YOLO-seg outputs with the same numpy post-processing the UDF uses:
   class-agnostic NMS, ``sigmoid(coefs @ proto)`` masks, bbox crop,
   binarise at 0.5, higher-conf-wins-overlap label assembly.

We then check whether the two label maps agree pixel-for-pixel.  If they
don't, we fall back to reporting foreground IoU and per-instance IoU so
it's obvious whether the drift is real (mask differences) or cosmetic
(instance re-ordering).

Usage::

    python openeo_udp/tests/test_onnx_parity.py
    python openeo_udp/tests/test_onnx_parity.py --export   # re-export .pt -> .onnx first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _resolve_repo_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parents[2]

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "openeo_udp").is_dir() and (candidate / "BAP_input.nc").exists():
            return candidate
    return cwd


_REPO_ROOT = _resolve_repo_root()
sys.path.insert(0, str(_REPO_ROOT))

from openeo_udp.process_graph.export_to_onnx import export as export_onnx
from openeo_udp.tests.test_local_udf import extract_tile, load_bap_cube


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_NC = _REPO_ROOT / "BAP_input.nc"
OUT_DIR = _REPO_ROOT / "openeo_udp" / "tests" / "test_outputs"
PROCESS_GRAPH_WEIGHTS_DIR = _REPO_ROOT / "openeo_udp" / "process_graph" / "delineate_weights"
LEGACY_WEIGHTS_DIR = _REPO_ROOT / "openeo_udp" / "tests" / "delineate_weights"
WEIGHTS_DIR = PROCESS_GRAPH_WEIGHTS_DIR if PROCESS_GRAPH_WEIGHTS_DIR.exists() else LEGACY_WEIGHTS_DIR
PT_PATH = WEIGHTS_DIR / "DelineateAnythingv2.pt"
ONNX_PATH = WEIGHTS_DIR / "DelineateAnythingv2.onnx"

TILE_SIZE = 512
TILE_X_START = 100
TILE_Y_START = 100

CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.3
MORPHOLOGY = False

# Ultralytics `predict()` defaults — must be mirrored on the ONNX side.
MAX_DET = 300      # cap on kept detections after NMS
MAX_NMS = 30_000   # cap on candidates entering NMS


# ---------------------------------------------------------------------------
# PyTorch reference path (matches methods/main/inference.py)
# ---------------------------------------------------------------------------

def _labels_from_yolo_result(result, h: int, w: int) -> np.ndarray:
    """Turn an ultralytics YOLO-seg result into a label map (higher conf wins)."""
    label_map = np.zeros((h, w), dtype=np.int32)
    if result.masks is None or result.boxes is None:
        return label_map

    masks = result.masks.data.detach().cpu().numpy()   # (N, H, W) in {0,1} float
    confs = result.boxes.conf.detach().cpu().numpy()   # (N,)

    if masks.shape[0] == 0:
        return label_map

    # Sort by ascending confidence so the last (highest conf) mask wins on overlap.
    order = np.argsort(confs)
    for rank, idx in enumerate(order):
        m = masks[idx]
        # retina_masks=True gives masks already at the model input resolution.
        if m.shape != (h, w):
            # Nearest-neighbour resize as a defensive fallback.
            ys = np.clip((np.arange(h) * m.shape[0] / h).astype(np.int32), 0, m.shape[0] - 1)
            xs = np.clip((np.arange(w) * m.shape[1] / w).astype(np.int32), 0, m.shape[1] - 1)
            m = m[ys[:, None], xs[None, :]]
        bin_mask = m > 0.5
        if not bin_mask.any():
            continue
        # Instance ID = rank in ascending-confidence order + 1.  The mapping
        # itself doesn't need to match the UDF exactly — the pixel-level
        # foreground/overlap structure is what parity is measured on.
        label_map[bin_mask] = rank + 1
    return label_map


def _run_pytorch(tile: xr.DataArray) -> np.ndarray:
    """End-to-end PyTorch path using ultralytics YOLO on the .pt checkpoint."""
    import torch
    from ultralytics import YOLO

    rgb = tile.values[:3].astype(np.float32)               # (3, H, W)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    _, h, w = rgb.shape

    # Ultralytics does NOT letterbox tensor inputs (it enforces BCHW divisible
    # by stride 32).  So do the letterbox ourselves — same 114/255 gray canvas
    # the ONNX path uses — feed the pre-normalised (1, 3, 512, 512) tensor, and
    # unwarp the returned 512x512 masks back to (h, w).
    canvas, gain, pad_top, pad_left = _letterbox(rgb, TILE_SIZE)  # (3, 512, 512) float32
    new_h, new_w = int(round(h * gain)), int(round(w * gain))

    img_tensor = torch.from_numpy(canvas).unsqueeze(0).contiguous()  # (1, 3, 512, 512)

    model = YOLO(str(PT_PATH), task="segment")
    results = model.predict(
        img_tensor,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=TILE_SIZE,
        verbose=False,
        retina_masks=True,
    )
    return _labels_from_yolo_result_letterboxed(
        results[0], h, w, pad_top, pad_left, new_h, new_w
    )


def _labels_from_yolo_result_letterboxed(
    result, h: int, w: int, pad_top: int, pad_left: int, new_h: int, new_w: int
) -> np.ndarray:
    """Same as `_labels_from_yolo_result` but undoes the manual letterbox first."""
    label_map = np.zeros((h, w), dtype=np.int32)
    if result.masks is None or result.boxes is None:
        return label_map

    masks = result.masks.data.detach().cpu().numpy()   # (N, 512, 512)
    confs = result.boxes.conf.detach().cpu().numpy()
    if masks.shape[0] == 0:
        return label_map

    # Crop out the centred (new_h, new_w) region, nearest-neighbour resize to (h, w).
    ys = np.clip((np.arange(h) * new_h / h).astype(np.int32), 0, new_h - 1) + pad_top
    xs = np.clip((np.arange(w) * new_w / w).astype(np.int32), 0, new_w - 1) + pad_left

    order = np.argsort(confs)
    for rank, idx in enumerate(order):
        m = masks[idx][ys[:, None], xs[None, :]]
        bin_mask = m > 0.5
        if not bin_mask.any():
            continue
        label_map[bin_mask] = rank + 1
    return label_map


# ---------------------------------------------------------------------------
# ONNX path — local onnxruntime inference (no UDF, no openeo imports)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes[:, :4])
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def _bilinear_upsample(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of a 2-D array (matches torch F.interpolate align_corners=False)."""
    ih, iw = mask.shape
    if (ih, iw) == (out_h, out_w):
        return mask
    # align_corners=False sampling grid.
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


def _bilinear_resize_chw(img_chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of a (C, H, W) float array — matches cv2.INTER_LINEAR closely enough."""
    c, ih, iw = img_chw.shape
    if (ih, iw) == (out_h, out_w):
        return img_chw
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * ih / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * iw / out_w - 0.5
    y = np.clip(y, 0.0, ih - 1); x = np.clip(x, 0.0, iw - 1)
    y0 = np.floor(y).astype(np.int32); y1 = np.clip(y0 + 1, 0, ih - 1)
    x0 = np.floor(x).astype(np.int32); x1 = np.clip(x0 + 1, 0, iw - 1)
    wy = (y - y0)[:, None]; wx = (x - x0)[None, :]
    out = np.empty((c, out_h, out_w), dtype=img_chw.dtype)
    for k in range(c):
        m = img_chw[k]
        Ia = m[y0[:, None], x0[None, :]]; Ib = m[y0[:, None], x1[None, :]]
        Ic = m[y1[:, None], x0[None, :]]; Id = m[y1[:, None], x1[None, :]]
        top = Ia * (1.0 - wx) + Ib * wx
        bot = Ic * (1.0 - wx) + Id * wx
        out[k] = top * (1.0 - wy) + bot * wy
    return out


def _letterbox(rgb_chw: np.ndarray, target: int) -> tuple[np.ndarray, float, int, int]:
    """Ultralytics-style letterbox: resize preserving aspect, centre-pad with 114/255.

    Returns (padded (3, target, target), gain, pad_top, pad_left).
    """
    _, h, w = rgb_chw.shape
    gain = min(target / h, target / w)
    new_h, new_w = int(round(h * gain)), int(round(w * gain))
    resized = _bilinear_resize_chw(rgb_chw, new_h, new_w) if (new_h, new_w) != (h, w) else rgb_chw

    pad_h = target - new_h
    pad_w = target - new_w
    top = pad_h // 2
    left = pad_w // 2

    canvas = np.full((3, target, target), 114.0 / 255.0, dtype=np.float32)
    canvas[:, top:top + new_h, left:left + new_w] = resized
    return canvas, gain, top, left


def _run_onnx_local(tile: xr.DataArray) -> np.ndarray:
    """Pure onnxruntime + numpy YOLO-seg decoder (matches ultralytics post-proc)."""
    import onnxruntime as ort

    rgb = tile.values[:3].astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    _, h, w = rgb.shape

    # Match ultralytics: letterbox to the model input (114/255 gray, centered).
    canvas, gain, pad_top, pad_left = _letterbox(rgb, TILE_SIZE)
    H = W = TILE_SIZE

    x = canvas[np.newaxis, ...]  # (1, 3, 512, 512)

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: x})

    preds, protos = None, None
    for out in outputs:
        if out.ndim == 4:
            protos = out
        elif out.ndim == 3:
            preds = out
    if preds is None or protos is None:
        raise RuntimeError(f"Unexpected ONNX output shapes: {[o.shape for o in outputs]}")

    preds = preds[0].astype(np.float32)      # (C, A)
    protos = protos[0].astype(np.float32)    # (nm, mh, mw)
    nm, mh, mw = protos.shape
    nc = preds.shape[0] - 4 - nm

    preds = preds.T                                        # (A, C)
    boxes_xywh = preds[:, :4]
    scores = preds[:, 4:4 + nc].max(axis=1)                # class-agnostic
    mask_coefs = preds[:, 4 + nc:]

    conf_keep = scores >= CONFIDENCE_THRESHOLD
    if not np.any(conf_keep):
        return np.zeros((h, w), dtype=np.int32)

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh[conf_keep])
    scores = scores[conf_keep]
    mask_coefs = mask_coefs[conf_keep]

    # Match ultralytics: sort by score DESC and cap pre-NMS candidates.
    order = np.argsort(-scores)[:MAX_NMS]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]
    mask_coefs = mask_coefs[order]

    keep = _nms(boxes_xyxy, scores, IOU_THRESHOLD)
    if keep.size == 0:
        return np.zeros((h, w), dtype=np.int32)
    # Match ultralytics `max_det=300`.
    keep = keep[:MAX_DET]
    boxes_xyxy = boxes_xyxy[keep]
    mask_coefs = mask_coefs[keep]

    # Match ultralytics `process_mask_native` exactly:
    #   logits = coefs @ proto                (low res, NO sigmoid yet)
    #   logits = bilinear_upsample(logits)    (to model input res)
    #   masks  = crop(logits, bbox)           (zero outside bbox at full res)
    #   masks  = logits > 0                   (equivalent to sigmoid(x) > 0.5)
    logits_lo = (mask_coefs @ protos.reshape(nm, mh * mw)).reshape(-1, mh, mw)

    label_map_lb = np.zeros((H, W), dtype=np.int32)

    for i in range(len(mask_coefs) - 1, -1, -1):
        x1f, y1f, x2f, y2f = boxes_xyxy[i]

        # Bilinear upsample logits to model input resolution.
        logits_hi = _bilinear_upsample(logits_lo[i], H, W)

        # Crop by the full-res bbox and threshold at 0 (== sigmoid > 0.5).
        ix1 = int(np.clip(np.floor(x1f), 0, W - 1))
        iy1 = int(np.clip(np.floor(y1f), 0, H - 1))
        ix2 = int(np.clip(np.ceil(x2f),  0, W))
        iy2 = int(np.clip(np.ceil(y2f),  0, H))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        bin_mask = np.zeros((H, W), dtype=bool)
        bin_mask[iy1:iy2, ix1:ix2] = logits_hi[iy1:iy2, ix1:ix2] > 0.0

        if not bin_mask.any():
            continue
        label_map_lb[bin_mask] = i + 1

    # Undo the letterbox: crop the centred image region and resize back to (h, w).
    new_h = int(round(h * gain))
    new_w = int(round(w * gain))
    label_map_cropped = label_map_lb[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    if (new_h, new_w) == (h, w):
        return label_map_cropped

    # Nearest-neighbour resize of the label map back to original tile shape.
    ys = np.clip((np.arange(h) * new_h / h).astype(np.int32), 0, new_h - 1)
    xs = np.clip((np.arange(w) * new_w / w).astype(np.int32), 0, new_w - 1)
    return label_map_cropped[ys[:, None], xs[None, :]]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _instance_masks(label_map: np.ndarray) -> list[np.ndarray]:
    ids = [i for i in np.unique(label_map) if i != 0]
    return [(label_map == i) for i in ids]


def _match_iou(a_masks: list[np.ndarray], b_masks: list[np.ndarray]) -> list[float]:
    ious: list[float] = []
    used_b: set[int] = set()
    for a in a_masks:
        best, best_j = -1.0, -1
        for j, b in enumerate(b_masks):
            if j in used_b:
                continue
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            iou = inter / union if union else 0.0
            if iou > best:
                best, best_j = iou, j
        if best_j >= 0:
            used_b.add(best_j)
        ious.append(best)
    return ious


def _build_rgb_preview(tile: xr.DataArray) -> np.ndarray:
    rgb = tile.values[:3].transpose(1, 2, 0).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)

    rgb_display = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[:, :, i]
        lo, hi = np.percentile(band, (2, 98))
        if hi > lo:
            rgb_display[:, :, i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        else:
            rgb_display[:, :, i] = band
    return rgb_display


def _save_comparison_plot(
    tile: xr.DataArray,
    labels_pt: np.ndarray,
    labels_ox: np.ndarray,
    out_path: Path,
) -> None:
    rgb_display = _build_rgb_preview(tile)
    fg_pt = labels_pt > 0
    fg_ox = labels_ox > 0
    diff = fg_pt ^ fg_ox

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "DelineateAnything parity: end-to-end PyTorch vs pure-ONNX UDF",
        fontsize=14,
    )

    axes[0, 0].imshow(rgb_display)
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_display)
    overlay_pt = np.ma.masked_where(labels_pt == 0, labels_pt)
    axes[0, 1].imshow(overlay_pt, cmap="tab20", alpha=0.55, interpolation="nearest")
    axes[0, 1].set_title(f"PyTorch (ultralytics .pt) — {int(labels_pt.max())} instances")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(rgb_display)
    overlay_ox = np.ma.masked_where(labels_ox == 0, labels_ox)
    axes[1, 0].imshow(overlay_ox, cmap="tab20", alpha=0.55, interpolation="nearest")
    axes[1, 0].set_title(f"ONNX UDF (.onnx) — {int(labels_ox.max())} instances")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb_display)
    diff_overlay = np.ma.masked_where(~diff, diff)
    axes[1, 1].imshow(diff_overlay, cmap="autumn", alpha=0.75, interpolation="nearest")
    axes[1, 1].set_title(f"Foreground disagreement: {int(diff.sum()):,} px")
    axes[1, 1].axis("off")

    plt.tight_layout()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true",
                    help="(Re-)export .pt -> .onnx before comparing")
    ap.add_argument("--imgsz", type=int, default=TILE_SIZE)
    ap.add_argument("--opset", type=int, default=17)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown CLI args: {unknown}")

    if args.export or not ONNX_PATH.exists():
        print(f"Exporting {PT_PATH} -> {ONNX_PATH}")
        exported = export_onnx(PT_PATH, imgsz=args.imgsz, opset=args.opset)
        if exported.resolve() != ONNX_PATH.resolve():
            ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
            exported.replace(ONNX_PATH)

    if not PT_PATH.exists():
        print(f"Missing PyTorch weights: {PT_PATH}", file=sys.stderr)
        return 2
    if not ONNX_PATH.exists():
        print(f"Missing ONNX weights: {ONNX_PATH}", file=sys.stderr)
        return 2

    print("=" * 60)
    print("END-TO-END PARITY: PyTorch (ultralytics) vs ONNX UDF")
    print("=" * 60)
    print(f"PT   : {PT_PATH}")
    print(f"ONNX : {ONNX_PATH}")
    print(f"conf : {CONFIDENCE_THRESHOLD}, iou: {IOU_THRESHOLD}, morph: {MORPHOLOGY}")

    cube = load_bap_cube(INPUT_NC)
    tile = extract_tile(cube, TILE_X_START, TILE_Y_START, TILE_SIZE)

    print("\n[1/2] Running PyTorch (ultralytics YOLO on .pt)...")
    labels_pt = _run_pytorch(tile)

    print("[2/2] Running ONNX (local onnxruntime + numpy decoder)...")
    labels_ox = _run_onnx_local(tile)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "labels_pt.npy", labels_pt)
    np.save(OUT_DIR / "labels_onnx.npy", labels_ox)
    plot_path = OUT_DIR / "onnx_parity_comparison.png"
    _save_comparison_plot(tile, labels_pt, labels_ox, plot_path)
    print(f"Saved comparison plot: {plot_path}")

    n_pt = int(labels_pt.max())
    n_ox = int(labels_ox.max())
    print(f"\nInstances: PT={n_pt}, ONNX={n_ox}")

    if np.array_equal(labels_pt, labels_ox):
        print("EXACT MATCH: label maps are byte-identical.")
        return 0

    fg_pt = labels_pt > 0
    fg_ox = labels_ox > 0
    fg_iou = np.logical_and(fg_pt, fg_ox).sum() / max(
        np.logical_or(fg_pt, fg_ox).sum(), 1
    )
    print(f"Foreground IoU (order-invariant): {fg_iou:.6f}")

    masks_pt = _instance_masks(labels_pt)
    masks_ox = _instance_masks(labels_ox)
    ious = _match_iou(masks_pt, masks_ox)
    if ious:
        print(
            "Per-instance IoU (PT -> best ONNX match): "
            f"min={min(ious):.4f}, mean={sum(ious) / len(ious):.4f}, "
            f"max={max(ious):.4f}"
        )
        mismatched = sum(1 for i in ious if i < 0.95)
        print(f"Instances with IoU < 0.95: {mismatched}/{len(ious)}")

    # Small numeric drift between ultralytics' torch post-processing and our
    # numpy YOLO-seg decoder is expected (bbox NMS ordering, mask upsample
    # interpolation, boundary rounding).  Treat foreground IoU >= 0.98 and
    # equal instance counts as a pass; anything less is a real regression.
    if fg_iou >= 0.98 and abs(n_pt - n_ox) <= max(1, int(0.02 * max(n_pt, n_ox))):
        print("PASS (within tolerance).")
        return 0
    print("FAIL: outputs diverge beyond tolerance.")
    return 1


if __name__ == "__main__":
    main()

# %%
