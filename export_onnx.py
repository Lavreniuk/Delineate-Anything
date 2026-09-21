#%%

#!/usr/bin/env python
"""Convert a local Delineate-Anything .pt checkpoint to ONNX.

Notebook workflow:
        from export_onnx import main
        main(validate=True)

        # Or explicit paths:
        main(
                weights_path=r"C:\Git_projects\Delineate-Anything\openeo_udp\process_graph\delineate_weights\DelineateAnythingv2.pt",
                output_path=r"C:\Git_projects\Delineate-Anything\openeo_udp\process_graph\delineate_weights\DelineateAnythingv2.onnx",
                validate=True,
        )
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_V2_WEIGHTS = (
    REPO / "openeo_udp" / "process_graph" / "delineate_weights" / "DelineateAnythingv2.pt"
)
DEFAULT_LOCAL_V2_ONNX = DEFAULT_LOCAL_V2_WEIGHTS.with_suffix(".onnx")


#%%


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


def main(
    weights_path: str | Path | None = None,
    output_path: str | Path | None = None,
    *,
    validate: bool = False,
    imgsz: int = 512,
    opset: int = 17,
    dynamic: bool = False,
    half: bool = False,
    simplify: bool = True,
    device: str = "cpu",
    verbose: bool = False,
) -> Path:
    """Notebook entrypoint: convert local .pt checkpoint to ONNX.

    If ``weights_path`` is omitted, uses the default local v2 checkpoint path.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    weights = Path(weights_path) if weights_path is not None else DEFAULT_LOCAL_V2_WEIGHTS
    if not weights.is_absolute():
        weights = (REPO / weights).resolve()
    if not weights.exists():
        # Backward-compat: old notebooks used openeo_udp/tests/delineate_weights.
        stale = Path("openeo_udp") / "tests" / "delineate_weights"
        fixed = Path("openeo_udp") / "process_graph" / "delineate_weights"
        if stale.as_posix() in weights.as_posix():
            remapped = Path(str(weights).replace(str(stale), str(fixed)))
            if remapped.exists():
                weights = remapped
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

    output = Path(output_path) if output_path is not None else DEFAULT_LOCAL_V2_ONNX
    if not output.is_absolute():
        output = (REPO / output).resolve()

    t0 = time.perf_counter()
    print(f"Weights:  {weights}")
    print(f"Output:   {output}")
    print(
        f"Exporting to ONNX (imgsz={imgsz}, opset={opset}, dynamic={dynamic}, half={half})..."
    )

    t_export = time.perf_counter()
    _export_onnx(
        weights_path=weights,
        output_path=output,
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        half=half,
        simplify=simplify,
        device=device,
    )
    print(f"Export finished in {time.perf_counter() - t_export:.1f}s")
    print(f"ONNX exported: {output} ({output.stat().st_size / 1e6:.1f} MB)")

    if validate:
        t_validate = time.perf_counter()
        _validate_onnx(weights, output, imgsz)
        print(f"Validation finished in {time.perf_counter() - t_validate:.1f}s")

    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")
    return output

# %%
