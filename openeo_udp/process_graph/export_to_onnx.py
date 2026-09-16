#%%
"""Export the Delineate-Anything YOLO checkpoint to ONNX.

Uses Ultralytics' built-in exporter, which produces a single ONNX file
containing the full YOLO-seg forward pass (backbone + detect head + proto
mask branch).  The ONNX file can then be loaded back through
``ultralytics.YOLO`` — ``AutoBackend`` transparently runs it via
``onnxruntime`` while reusing the exact same Python-side post-processing
(NMS, mask assembly, ``retina_masks``) as the PyTorch path.

That means the openEO UDF (``openeo_udp/udf/delineate_pytorch.py``) does
not need to change: point ``weights_url`` at the ``.onnx`` file and swap
``torch`` for ``onnxruntime`` in the ``udf-dependency-archives``.

Usage::

    python openeo_udp/tests/export_to_onnx.py \\
        --pt openeo_udp/tests/delineate_weights/DelineateAnything-S.pt \\
        --imgsz 512 --opset 17
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

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

from ultralytics import YOLO


def export(pt_path: Path, imgsz: int = 512, opset: int = 17,
           dynamic: bool = False, simplify: bool = False,
           half: bool = False) -> Path:
    """Export ``pt_path`` to ONNX next to it. Returns the ONNX path."""
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    model = YOLO(str(pt_path), task="segment")
    # Ultralytics returns the exported file path (str).  Options kept
    # conservative so the graph stays 1:1 with the PyTorch forward pass:
    #   * dynamic=False  → fixed 1x3ximgszximgsz input, no shape ops
    #   * simplify=False → no onnxsim graph rewrites
    #   * half=False     → keep fp32 to preserve numeric parity
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
    )
    out_path = Path(out).resolve()
    print(f"Exported ONNX: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def package_model_zip(onnx_path: Path, zip_path: Path, mount_dir: str = "onnx_models") -> Path:
    """Create a dependency archive with ONNX at <mount_dir>/<model>.onnx."""
    onnx_path = onnx_path.resolve()
    zip_path = zip_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    arcname = f"{mount_dir.strip('/')}/{onnx_path.name}"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, arcname=arcname)

    print(f"Packaged model archive: {zip_path} (contains: {arcname})")
    return zip_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export DelineateAnything YOLO .pt to ONNX")
    ap.add_argument(
        "--pt",
        type=Path,
        default=_REPO_ROOT / "openeo_udp" / "tests" / "delineate_weights"
        / "DelineateAnything-S.pt",
        help="Path to the .pt checkpoint",
    )
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true",
                    help="Export with dynamic batch/spatial axes")
    ap.add_argument("--simplify", action="store_true",
                    help="Run onnxsim on the exported graph")
    ap.add_argument("--half", action="store_true",
                    help="Export in fp16 (breaks exact parity with fp32 PyTorch)")
    ap.add_argument("--package-zip", action="store_true",
                    help="Also create a dependency archive with onnx_models/<model>.onnx")
    ap.add_argument("--zip-path", type=Path, default=None,
                    help="Path for the model archive zip (default: sibling .zip next to .onnx)")
    ap.add_argument("--mount-dir", type=str, default="onnx_models",
                    help="Folder name used as archive fragment mount target")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown CLI args: {unknown}")

    onnx_path = export(
        pt_path=args.pt.resolve(),
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
    )

    if args.package_zip:
        default_zip = onnx_path.with_suffix(".zip")
        zip_path = args.zip_path.resolve() if args.zip_path else default_zip
        package_model_zip(onnx_path=onnx_path, zip_path=zip_path, mount_dir=args.mount_dir)


if __name__ == "__main__":
    main()

# %%
