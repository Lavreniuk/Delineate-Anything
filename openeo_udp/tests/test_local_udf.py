#%%
"""Local UDF test: run the Delineate-Anything PyTorch UDF on BAP_input.nc.

Loads the local BAP composite NetCDF, distills it into an xarray DataArray,
feeds a 512×512 tile straight into
``openeo_udp.udf.delineate_pytorch.apply_datacube``, and saves a diagnostic
plot to ``openeo_udp/tests/test_outputs/inference_output.png``.

Requirements (installed in the local Python env, not via the openEO deps
archive)::

    pip install torch ultralytics xarray netCDF4 matplotlib numpy openeo

The .pt checkpoint is downloaded from Hugging Face on first run and cached
under ``delineate_weights/`` in the repo root.

Usage::

    python openeo_udp/tests/test_local_udf.py
"""

from __future__ import annotations

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

from openeo_udp.udf.delineate_onnx import apply_datacube
from openeo.udf import XarrayDataCube

# ---------------------------------------------------------------------------
# Config — tweak here
# ---------------------------------------------------------------------------
INPUT_NC = _REPO_ROOT / "BAP_input.nc"
OUT_DIR = _REPO_ROOT / "openeo_udp" / "tests" / "test_outputs"

TILE_SIZE = 512
TILE_X_START = 100
TILE_Y_START = 100

# UDF context. `input_scale` divides raw reflectance to bring it into [0, 1].
# 3000 matches the S2 BOA scaling used by the reference pipeline. Set to 1.0
# if your input is already in [0, 1].
CONTEXT = {
    "confidence_threshold": 0.005,
    "iou_threshold": 0.3,
    "morphology": False,
    "input_scale": 1.0,
    "weights_url": "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything-S.pt",
}   


# ---------------------------------------------------------------------------
# Load BAP_input.nc → (bands, y, x) DataArray in raw reflectance
# ---------------------------------------------------------------------------

def load_bap_cube(path: Path) -> xr.DataArray:
    """Return an (bands, y, x) DataArray with the RGB bands from a BAP NetCDF."""
    ds = xr.open_dataset(path)
    print(f"Variables : {list(ds.data_vars)}")
    print(f"Dims      : {dict(ds.sizes)}")
    print(f"Coords    : {list(ds.coords)}")

    # Non-band variables that BAP/openEO NetCDF outputs typically include
    # as CF grid-mapping / auxiliary coordinates — skip them when stacking.
    _NON_BAND_VARS = {"crs"}

    if "bands" in ds.dims:
        # Already stacked as (bands, y, x) inside a single data var.
        var_name = next(v for v in ds.data_vars if v not in _NON_BAND_VARS)
        da = ds[var_name]
    else:
        # Each band is its own data_var — stack the spatial ones into `bands`.
        # A real band has both y and x dims; `crs` etc. are scalars → filtered.
        band_names = [
            v for v in ds.data_vars
            if v not in _NON_BAND_VARS
            and np.issubdtype(ds[v].dtype, np.number)
            and {"y", "x"}.issubset(set(ds[v].dims))
        ]
        if not band_names:
            raise ValueError(f"No band-like variables found in {path}")
        da = xr.concat([ds[v] for v in band_names], dim="bands")
        da = da.assign_coords(bands=band_names)

    # Collapse any singleton / multi time dim.
    t_dim = next((d for d in da.dims if d in ("t", "time")), None)
    if t_dim is not None:
        da = da.squeeze(t_dim) if da.sizes[t_dim] == 1 else da.mean(dim=t_dim)

    da = da.astype(np.float32)

    print(f"Cube shape: {da.shape}, dims={list(da.dims)}, dtype={da.dtype}")
    if "bands" in da.dims:
        for i, name in enumerate(da.coords["bands"].values):
            v = da.isel(bands=i).values
            valid = v[~np.isnan(v)]
            if valid.size:
                print(
                    f"  band {name}: min={valid.min():.1f}, "
                    f"max={valid.max():.1f}, mean={valid.mean():.1f}"
                )
    return da


def extract_tile(da: xr.DataArray, x0: int, y0: int, size: int) -> xr.DataArray:
    dims = list(da.dims)
    y_dim = next(d for d in dims if d in ("y", "lat", "latitude"))
    x_dim = next(d for d in dims if d in ("x", "lon", "longitude"))
    return da.isel({y_dim: slice(y0, y0 + size), x_dim: slice(x0, x0 + size)})


def main() -> None:
    print("=" * 60)
    print("LOCAL PYTORCH UDF TEST")
    print("=" * 60)
    print(f"Loading {INPUT_NC}")

    cube = load_bap_cube(INPUT_NC)
    tile = extract_tile(cube, TILE_X_START, TILE_Y_START, TILE_SIZE)
    print(f"\nTile: shape={tile.shape}, dims={list(tile.dims)}")

    print("\nRunning UDF...")
    result = apply_datacube(tile, CONTEXT)
    # The UDF returns an XarrayDataCube; unwrap to a DataArray for downstream use.
    result_da = result.get_array() if hasattr(result, "get_array") else result
    labels = result_da.isel(bands=0).values.astype(np.int32)
    n_fields = int(labels.max())
    print(
        f"Detected {n_fields} fields, "
        f"foreground = {100 * (labels > 0).mean():.2f}% of pixels"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # For display: divide by the same input_scale used inside the UDF, then do
    # a per-band 2–98 percentile stretch so the preview isn't near-black.
    rgb = tile.values[:3].transpose(1, 2, 0).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0) / float(CONTEXT.get("input_scale", 3000.0))
    rgb = np.clip(rgb, 0.0, 1.0)

    rgb_display = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[:, :, i]
        lo, hi = np.percentile(band, (2, 98))
        if hi > lo:
            rgb_display[:, :, i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        else:
            rgb_display[:, :, i] = band

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Delineate-Anything (PyTorch UDF) — {n_fields} fields", fontsize=13)

    axes[0].imshow(rgb_display)
    axes[0].set_title("Input RGB (2–98% stretch for display)")
    axes[0].axis("off")

    axes[1].imshow(rgb_display)
    overlay = np.ma.masked_where(labels == 0, labels)
    axes[1].imshow(overlay, cmap="tab20", alpha=0.55, interpolation="nearest")
    axes[1].set_title("Instances overlaid on input")
    axes[1].axis("off")

    plt.tight_layout()
    out_png = OUT_DIR / "inference_output.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")
    plt.show()


if __name__ == "__main__":
    main()

# %%
