#!/usr/bin/env python
"""Export the Delineate-Anything YOLO field-boundary model to ONNX.

Delineate-Anything (https://huggingface.co/MykolaL/DelineateAnything) is an
Ultralytics YOLO instance-segmentation model. Ultralytics provides a built-in
ONNX exporter, so this script:

  1. Downloads (or accepts a local path to) the .pt checkpoint.
  2. Loads it with `ultralytics.YOLO(...)`.
  3. Exports to ONNX with sensible defaults for tiled inference
     (imgsz=512, opset 17, optional dynamic batch / spatial axes).
  4. Optionally runs an `onnxruntime` parity check vs. the PyTorch model.

Usage:
    python export_onnx.py --variant large --output DelineateAnything.onnx --validate

    # Or from a local .pt file:
    python export_onnx.py --weights ./DelineateAnything.pt --output DelineateAnything.onnx

Requirements (local environment):
    pip install ultralytics onnx onnxruntime onnxslim huggingface_hub

Notes:
- Ultralytics writes the ONNX file next to the .pt file by default. We move it
  to --output afterwards so paths follow this repo's release layout.
- `--dynamic` enables dynamic batch + spatial axes. Some downstream runtimes
  (notably older onnxruntime versions on the openEO backend) prefer a fixed
  shape; in that case omit --dynamic and the model is exported at
  (1, 3, imgsz, imgsz).
- The exported graph still emits raw YOLO seg outputs (detections + mask
  prototypes). Post-processing (NMS, mask assembly, polygonization) is NOT
  part of the ONNX file and must be implemented in the consuming UDF.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)

HF_REPO_ID = "MykolaL/DelineateAnything"
VARIANTS = {
    "small": "DelineateAnything-S.pt",
    "large": "DelineateAnything.pt",
}


def _resolve_weights(variant: str | None, weights_arg: str | None) -> Path:
    """Return a local path to the .pt checkpoint, downloading from HF if needed."""
    if weights_arg:
        p = Path(weights_arg)
        if not p.is_absolute():
            p = (REPO / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p

    if variant is None:
        raise ValueError("Provide either --weights or --variant {small,large}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choices: {list(VARIANTS)}")

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: pip install huggingface_hub"
        ) from e

    logger.info("Downloading %s from %s ...", VARIANTS[variant], HF_REPO_ID)
    local = hf_hub_download(repo_id=HF_REPO_ID, filename=VARIANTS[variant])
    return Path(local)


def _export_onnx(
    weights_path: Path,
    output_path: Path,
    imgsz: int,
    opset: int,
    dynamic: bool,
    half: bool,
    simplify: bool,
    device: str,
) -> Path:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing export dependency. Install with: pip install ultralytics"
        ) from e

    model = YOLO(str(weights_path))
    logger.info("Loaded YOLO model task=%s from %s", getattr(model, "task", "?"), weights_path)

    # Ultralytics writes the .onnx alongside the .pt file and returns its path.
    onnx_tmp = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
        device=device,
    )
    onnx_tmp = Path(onnx_tmp)
    if not onnx_tmp.exists():
        raise RuntimeError(f"Ultralytics reported export at {onnx_tmp}, but file is missing.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if onnx_tmp.resolve() != output_path.resolve():
        shutil.move(str(onnx_tmp), str(output_path))
    return output_path


def _validate_onnx(weights_path: Path, onnx_path: Path, imgsz: int) -> None:
    import numpy as np

    try:
        import onnx
        import onnxruntime as ort
        import torch
        from ultralytics import YOLO
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing validation dependency. Install: pip install onnx onnxruntime torch ultralytics"
        ) from e

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Build a small RGB tile and compare raw network outputs.
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)

    # Torch reference forward (no post-processing, just raw model() outputs).
    yolo = YOLO(str(weights_path))
    torch_model = yolo.model.float().eval()
    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x))

    # Flatten torch outputs to a list of numpy arrays for shape reporting.
    def _flatten(obj):
        if isinstance(obj, (list, tuple)):
            out = []
            for o in obj:
                out.extend(_flatten(o))
            return out
        return [obj]

    torch_arrays = [t.detach().cpu().numpy() for t in _flatten(y_torch) if hasattr(t, "detach")]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_outputs = sess.run(None, {input_name: x})

    print("Validation OK")
    print(f"  ONNX inputs : {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
    print(f"  ONNX outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")
    print(f"  Torch raw output shapes: {[a.shape for a in torch_arrays]}")

    # Best-effort numeric parity on the first output that matches shape.
    for i, oarr in enumerate(onnx_outputs):
        match = next((t for t in torch_arrays if t.shape == oarr.shape), None)
        if match is not None:
            diff = float(np.max(np.abs(match - oarr)))
            print(f"  Output[{i}] shape={oarr.shape} max_abs_diff={diff:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", required=True, help="Output ONNX path (absolute or repo-relative)")
    parser.add_argument("--variant", choices=list(VARIANTS), default=None,
                        help="Which DelineateAnything checkpoint to fetch from Hugging Face")
    parser.add_argument("--weights", default=None,
                        help="Optional local path to a .pt checkpoint (overrides --variant)")
    parser.add_argument("--imgsz", type=int, default=512, help="Export input size (default 512, matches training tile)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default 17)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Export with dynamic batch + spatial axes (default: fixed 1x3xHxW)")
    parser.add_argument("--half", action="store_true", help="Export FP16 weights (requires --device cuda)")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnxslim graph simplification")
    parser.add_argument("--device", default="cpu", help="Device for the export forward pass (cpu / 0 / cuda)")
    parser.add_argument("--validate", action="store_true",
                        help="Run onnx.checker + onnxruntime parity check after export")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (REPO / output_path).resolve()

    t0 = time.perf_counter()
    weights_path = _resolve_weights(args.variant, args.weights)

    print(f"Weights:  {weights_path}")
    print(f"Output:   {output_path}")
    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset}, dynamic={args.dynamic}, half={args.half})...")

    t_export = time.perf_counter()
    _export_onnx(
        weights_path=weights_path,
        output_path=output_path,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
        simplify=not args.no_simplify,
        device=args.device,
    )
    print(f"Export finished in {time.perf_counter() - t_export:.1f}s")
    print(f"ONNX exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    if args.validate:
        t_validate = time.perf_counter()
        _validate_onnx(weights_path, output_path, args.imgsz)
        print(f"Validation finished in {time.perf_counter() - t_validate:.1f}s")

    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
