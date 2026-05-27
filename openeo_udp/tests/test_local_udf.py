#%%

"""
Local UDF testing: run the Delineate-Anything UDFs on a local NetCDF cube.

This script loads BAP_input.nc and applies the inference (and optionally
post-processing) UDF locally, so you can inspect intermediate results
without submitting an openEO job.

Usage:
    python openeo_udp/tests/test_local_udf.py
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_NC = _REPO_ROOT / "BAP_input.nc"
TILE_SIZE = 512  # model input size

# Which tile to test (top-left pixel coords in the full raster)
# Try different positions if the first tile is mostly nodata
TILE_X_START = 100
TILE_Y_START = 100

# Set to True to also run post-processing
RUN_POSTPROC = True

# --- Tuneable parameters (play with these!) ---
CONFIDENCE_THRESHOLD = 0.5   # YOLO detection confidence (lower = more detections)
MASK_THRESHOLD = 0.3      # Binary mask threshold (lower = more mask area)
MIN_AREA_PX = 10       # Min field size in pixels (lower = keep smaller fields)
MIN_HOLE_AREA_PX = 10        # Min hole area to preserve (lower = fill more holes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cube(path: Path) -> xr.DataArray:
    """Load a NetCDF and return as xr.DataArray suitable for UDF input."""
    ds = xr.open_dataset(path)
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dataset dims: {dict(ds.dims)}")
    print(f"Dataset coords: {list(ds.coords)}")

    # Try to build a DataArray from the dataset
    # The BAP output should have bands as variables or a 'bands' dim
    if "bands" in ds.dims:
        # Already has a bands dimension - pick the first data var
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
    else:
        # Variables are the bands - stack them (skip non-numeric like 'crs')
        band_names = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        arrays = [ds[v] for v in band_names]
        da = xr.concat(arrays, dim="bands")
        da["bands"] = band_names

    print(f"\nDataArray shape: {da.shape}")
    print(f"DataArray dims: {list(da.dims)}")
    print(f"DataArray dtype: {da.dtype}")

    # Print per-band stats
    if "bands" in da.dims:
        for i, b in enumerate(da.coords["bands"].values):
            band_data = da.isel(bands=i).values.astype(np.float32)
            valid = band_data[~np.isnan(band_data)]
            if len(valid) > 0:
                print(f"  {b}: min={valid.min():.2f}, max={valid.max():.2f}, "
                      f"mean={valid.mean():.2f}, nan%={100*(1-len(valid)/band_data.size):.1f}")
            else:
                print(f"  {b}: ALL NaN")

    return da


def extract_tile(da: xr.DataArray, x_start: int, y_start: int, size: int) -> xr.DataArray:
    """Extract a tile from the datacube."""
    dims = list(da.dims)

    # Find spatial dims
    y_dim = next((d for d in dims if d in ("y", "lat", "latitude")), None)
    x_dim = next((d for d in dims if d in ("x", "lon", "longitude")), None)

    if y_dim is None or x_dim is None:
        # Fallback: assume last two dims are spatial
        spatial_dims = [d for d in dims if d != "bands" and d != "t" and d != "time"]
        y_dim, x_dim = spatial_dims[0], spatial_dims[1]

    ny = da.sizes[y_dim]
    nx = da.sizes[x_dim]

    y_end = min(y_start + size, ny)
    x_end = min(x_start + size, nx)

    tile = da.isel({y_dim: slice(y_start, y_end), x_dim: slice(x_start, x_end)})
    print(f"\nExtracted tile: [{y_start}:{y_end}, {x_start}:{x_end}] -> shape {tile.shape}")
    return tile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


print("=" * 60)
print("LOCAL UDF TEST - Delineate-Anything")
print("=" * 60)

# Load input
print(f"\nLoading: {INPUT_NC}")
da = load_cube(INPUT_NC)

# Extract a tile
tile = extract_tile(da, TILE_X_START, TILE_Y_START, TILE_SIZE)

# Drop time dim if present
dims = list(tile.dims)
t_dim = next((d for d in dims if d in ("t", "time")), None)
if t_dim is not None:
    if tile.sizes[t_dim] == 1:
        tile = tile.squeeze(t_dim)
    else:
        print(f"  Reducing time dim '{t_dim}' ({tile.sizes[t_dim]} steps) via mean")
        tile = tile.mean(dim=t_dim)

print(f"\nTile ready for UDF: dims={list(tile.dims)}, shape={tile.shape}")

# -----------------------------------------------------------------------
# Run inference UDF
# -----------------------------------------------------------------------
print("\n" + "-" * 60)
print("RUNNING INFERENCE UDF")
print("-" * 60)

# Monkey-patch the MODEL_DIR so the UDF finds the local model
import openeo_udp.udf.delineate_inference as _inf_mod
_inf_mod.MODEL_DIR = str(_REPO_ROOT)

# Pad tile to 512x512 if smaller (model requires fixed input size)
dims = list(tile.dims)
b_dim = next((d for d in dims if d in ("bands", "band", "spectral")), None)
spatial_dims = [d for d in dims if d != b_dim]
y_dim_t, x_dim_t = spatial_dims[0], spatial_dims[1]
h_tile, w_tile = tile.sizes[y_dim_t], tile.sizes[x_dim_t]

if h_tile < TILE_SIZE or w_tile < TILE_SIZE:
    pad_h = max(0, TILE_SIZE - h_tile)
    pad_w = max(0, TILE_SIZE - w_tile)
    print(f"  Padding tile from ({h_tile}, {w_tile}) to ({h_tile+pad_h}, {w_tile+pad_w})")
    # Pad with zeros
    pad_widths = {d: (0, 0) for d in dims}
    pad_widths[y_dim_t] = (0, pad_h)
    pad_widths[x_dim_t] = (0, pad_w)
    tile = tile.pad(pad_widths, mode="constant", constant_values=0)

from openeo_udp.udf.delineate_inference import apply_datacube as inference_udf

# If data is already in [0, 1] (e.g. from build_bap_only), skip scaling.
# If data is raw reflectance (e.g. 0-10000), divide by LINEAR_SCALE_MAX.
tile_f32 = tile.astype(np.float32)
tile_scaled = tile_f32.clip(0, 1)

context = {
    "confidence_threshold": CONFIDENCE_THRESHOLD,
}

result = inference_udf(tile_scaled, context)

# Crop back to original size
if h_tile < TILE_SIZE or w_tile < TILE_SIZE:
    result = result.isel({y_dim_t: slice(0, h_tile), x_dim_t: slice(0, w_tile)})

print(f"\nInference output: dims={list(result.dims)}, shape={result.shape}")
print(f"  dtype: {result.dtype}")

# Print per-band stats of the result
if "bands" in result.dims:
    band_names = ["red", "green", "blue", "detection", "mask"]
    for i in range(result.shape[0]):
        name = band_names[i] if i < len(band_names) else f"band_{i}"
        vals = result.isel(bands=i).values
        print(f"  {name}: min={vals.min():.4f}, max={vals.max():.4f}, "
                f"mean={vals.mean():.4f}, nonzero%={100*(vals!=0).sum()/vals.size:.1f}")
else:
    vals = result.values
    print(f"  min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

# -----------------------------------------------------------------------
# Run post-processing UDF (optional)
# -----------------------------------------------------------------------
if RUN_POSTPROC:
    print("\n" + "-" * 60)
    print("RUNNING POST-PROCESSING UDF")
    print("-" * 60)

    from openeo_udp.udf.delineate_postprocess import apply_datacube as postproc_udf

    postproc_context = {
        "mask_threshold": MASK_THRESHOLD,
        "min_area_px": MIN_AREA_PX,
        "min_hole_area_px": MIN_HOLE_AREA_PX,
    }

    postproc_result = postproc_udf(result, postproc_context)

    print(f"\nPost-proc output: dims={list(postproc_result.dims)}, shape={postproc_result.shape}")

    # Instance stats
    if "bands" in postproc_result.dims:
        for i in range(postproc_result.shape[0]):
            vals = postproc_result.isel(bands=i).values
            n_unique = len(np.unique(vals[vals > 0]))
            print(f"  band {i}: min={vals.min():.0f}, max={vals.max():.0f}, "
                    f"unique_instances={n_unique}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

OUT_DIR = _REPO_ROOT / "openeo_udp" / "tests" / "test_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the unpadded tile for input display (already [0,1])
input_rgb = tile.values[:3].transpose(1, 2, 0).astype(np.float32)
input_rgb = np.nan_to_num(input_rgb, nan=0.0)
input_display = np.clip(input_rgb, 0, 1)

# --- Plot 1: Inference UDF output (2x2) ---
if "bands" in result.dims:
    # Crop result to original (unpadded) size for display
    res_cropped = result.isel({y_dim_t: slice(0, h_tile), x_dim_t: slice(0, w_tile)}) \
        if (h_tile < TILE_SIZE or w_tile < TILE_SIZE) else result

    udf_rgb = res_cropped.values[:3].transpose(1, 2, 0)
    det = res_cropped.isel(bands=3).values
    mask = res_cropped.isel(bands=4).values

    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 12))
    fig1.suptitle("INFERENCE UDF OUTPUT", fontsize=14)

    # Top-left: RGB as fed to model
    axes1[0, 0].imshow(np.clip(udf_rgb, 0, 1))
    axes1[0, 0].set_title("RGB [0,1] (model input)")
    axes1[0, 0].axis("off")

    # Top-right: Detection confidence
    im_det = axes1[0, 1].imshow(det, cmap="hot")
    axes1[0, 1].set_title(f"Detection\nmax={det.max():.4f}, >0: {100*(det>0).mean():.1f}%")
    axes1[0, 1].axis("off")
    plt.colorbar(im_det, ax=axes1[0, 1], fraction=0.046)

    # Bottom-left: Mask probability
    im_mask = axes1[1, 0].imshow(mask, cmap="viridis")
    axes1[1, 0].set_title(f"Mask probability\nmax={mask.max():.4f}, >0: {100*(mask>0).mean():.1f}%")
    axes1[1, 0].axis("off")
    plt.colorbar(im_mask, ax=axes1[1, 0], fraction=0.046)

    # Bottom-right: Mask overlay on input RGB
    axes1[1, 1].imshow(input_display)
    axes1[1, 1].imshow(mask, cmap="Reds", alpha=0.5 * (mask > 0).astype(float))
    axes1[1, 1].set_title("Mask overlay on input")
    axes1[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "02_inference_output.png"), dpi=150)
    plt.show()

# --- Plot 3: Post-processing output (2x2) ---
if RUN_POSTPROC and 'postproc_result' in dir():
    # Bands: mask_probability, binary_mask, instances
    mask_prob_out = postproc_result.isel(bands=0).values
    binary_mask   = postproc_result.isel(bands=1).values
    instances     = postproc_result.isel(bands=2).values
    n_fields = int(instances.max())

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
    fig2.suptitle(f"POST-PROCESSING OUTPUT ({n_fields} fields)", fontsize=14)

    # Top-left: mask probability (pass-through from inference)
    im_mp = axes2[0, 0].imshow(mask_prob_out, cmap="viridis")
    axes2[0, 0].set_title(f"Mask probability\nmax={mask_prob_out.max():.4f}")
    axes2[0, 0].axis("off")
    plt.colorbar(im_mp, ax=axes2[0, 0], fraction=0.046)

    # Top-right: Binary mask
    axes2[0, 1].imshow(binary_mask, cmap="gray")
    axes2[0, 1].set_title(f"Binary mask (threshold={MASK_THRESHOLD})")
    axes2[0, 1].axis("off")

    # Bottom-left: Instance labels
    im_inst = axes2[1, 0].imshow(instances, cmap="tab20", interpolation="nearest")
    axes2[1, 0].set_title(f"Instance labels ({n_fields} fields)")
    axes2[1, 0].axis("off")
    plt.colorbar(im_inst, ax=axes2[1, 0], fraction=0.046)

    # Bottom-right: Instances overlaid on input RGB
    axes2[1, 1].imshow(input_display)
    inst_masked = np.ma.masked_where(instances == 0, instances)
    axes2[1, 1].imshow(inst_masked, cmap="tab20", alpha=0.6, interpolation="nearest")
    axes2[1, 1].set_title("Instances overlay on input")
    axes2[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "03_postproc_output.png"), dpi=150)
    plt.show()

# %%
