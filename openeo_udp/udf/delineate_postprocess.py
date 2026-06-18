"""OpenEO UDF: Delineate-Anything post-processing (mask → polygons/instances).

This UDF operates on the OUTPUT of the inference UDF (detection + mask bands)
at a larger spatial extent (multiple inference tiles) to:

  1. Threshold the mask probability to produce a binary field mask.
  2. Label connected components → individual field instances.
  3. Filter out small fields below a minimum area threshold.
  4. Output an integer instance map (each field gets a unique ID).

This UDF should be called with a LARGER apply_neighborhood window than the
inference UDF (e.g. 2048×2048 inner with 64px overlap) so that field merging
across inference tile boundaries happens naturally via connected components.

Invocation
----------
Call via ``apply_neighborhood`` on the inference output datacube, selecting
only the "mask" band.

Input: (bands=1, y, x) — the mask probability from inference UDF.
Output: (bands=1, y, x) — integer instance labels (0 = background).

Context overrides::

    {
        "mask_threshold": 0.5,         # binarization threshold
        "min_area_px": 50,             # minimum field area in pixels
        "min_hole_area_px": 25,        # minimum hole area to keep
    }
"""

import logging

import numpy as np
import xarray as xr
from scipy import ndimage

from openeo.metadata import CollectionMetadata

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_MIN_AREA_PX = 50
DEFAULT_MIN_HOLE_AREA_PX = 25


# ===========================================================================
# Post-processing logic
# ===========================================================================

def _threshold_and_label(
    mask_prob: np.ndarray,
    threshold: float,
    min_area_px: int,
    min_hole_area_px: int,
) -> np.ndarray:
    """Threshold mask probabilities and produce labelled instances.

    Parameters
    ----------
    mask_prob : (H, W) float32, values in [0, 1]
    threshold : binarization threshold
    min_area_px : minimum field area in pixels (smaller → removed)
    min_hole_area_px : minimum hole area to keep (smaller → filled)

    Returns
    -------
    (H, W) int32 instance label map (0 = background)
    """
    h, w = mask_prob.shape
    logger.info("Post-processing: shape=(%d, %d), threshold=%.3f, "
                "min_area=%d, min_hole=%d",
                h, w, threshold, min_area_px, min_hole_area_px)

    # Binarize
    binary = (mask_prob > threshold).astype(np.uint8)
    logger.info("Binary mask: %.1f%% foreground", 100.0 * binary.mean())

    if binary.max() == 0:
        logger.info("No fields detected — returning empty instance map")
        return np.zeros((h, w), dtype=np.int32)

    # Fill small holes in the binary mask
    if min_hole_area_px > 0:
        holes = 1 - binary
        hole_labels, n_holes = ndimage.label(holes)
        hole_sizes = ndimage.sum(holes, hole_labels, range(1, n_holes + 1))
        for i, size in enumerate(hole_sizes, start=1):
            if size < min_hole_area_px:
                binary[hole_labels == i] = 1
        logger.info("Filled %d small holes (< %d px)",
                    int((hole_sizes < min_hole_area_px).sum()), min_hole_area_px)

    # Label connected components (4-connectivity for cleaner boundaries)
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connected
    labels, n_features = ndimage.label(binary, structure=structure)
    logger.info("Connected components: %d features", n_features)

    if n_features == 0:
        return np.zeros((h, w), dtype=np.int32)

    # Filter by area
    component_sizes = ndimage.sum(binary, labels, range(1, n_features + 1))
    keep_mask = component_sizes >= min_area_px
    n_kept = int(keep_mask.sum())
    n_removed = n_features - n_kept
    logger.info("Filtering: keeping %d / %d fields (removed %d < %d px)",
                n_kept, n_features, n_removed, min_area_px)

    if n_kept == 0:
        return np.zeros((h, w), dtype=np.int32)

    # Relabel: remove small components, renumber sequentially
    # Build lookup: old_label → new_label (0 for removed)
    lut = np.zeros(n_features + 1, dtype=np.int32)
    new_id = 1
    for old_id in range(1, n_features + 1):
        if keep_mask[old_id - 1]:
            lut[old_id] = new_id
            new_id += 1

    instances = lut[labels]
    logger.info("Final instance map: %d fields, max_id=%d",
                n_kept, int(instances.max()))

    return instances


# ===========================================================================
# OpenEO UDF entry points
# ===========================================================================

def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    """Declare 3-band output (mask_probability, binary_mask, instances)."""
    return metadata.rename_labels(
        dimension="bands",
        target=["mask_probability", "binary_mask", "instances"],
    )


def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    """Post-processing UDF entry point.

    Input: (bands, y, x) with at least the "mask" band from inference.
    Output: (bands=1, y, x) with integer field instance labels.
    """
    threshold = float(context.get("mask_threshold", DEFAULT_MASK_THRESHOLD))
    min_area_px = int(context.get("min_area_px", DEFAULT_MIN_AREA_PX))
    min_hole_area_px = int(context.get("min_hole_area_px", DEFAULT_MIN_HOLE_AREA_PX))

    logger.info("=== Delineate Post-processing UDF START ===")
    logger.info("Input cube dims: %s, shape: %s", list(cube.dims), cube.shape)

    dims = list(cube.dims)

    # Handle time dimension
    t_dim = next((d for d in dims if d in ("t", "time")), None)
    if t_dim is not None:
        cube = cube.squeeze(t_dim) if cube.sizes[t_dim] == 1 else cube.mean(dim=t_dim)
        dims = list(cube.dims)

    # Identify dimensions
    b_dim = next((d for d in dims if d in ("bands", "band", "spectral")), None)
    if b_dim is None:
        # No band dim — treat entire cube as single mask
        spatial_dims = dims
        y_dim, x_dim = spatial_dims[0], spatial_dims[1]
        mask_prob = cube.values.astype(np.float32)
        rgb_bands = None
    else:
        spatial_dims = [d for d in dims if d != b_dim]
        y_dim, x_dim = spatial_dims[0], spatial_dims[1]
        data = cube.transpose(b_dim, y_dim, x_dim).values

        # Inference outputs 5 bands: red, green, blue, detection, mask
        # Extract RGB (first 3) and mask (last band)
        rgb_bands = data[:3, :, :] if data.shape[0] >= 5 else None
        mask_prob = data[-1, :, :].astype(np.float32)

    logger.info("Mask prob stats: min=%.4f, max=%.4f, mean=%.4f, shape=%s",
                float(mask_prob.min()), float(mask_prob.max()),
                float(mask_prob.mean()), mask_prob.shape)

    # Replace NaN
    mask_prob = np.nan_to_num(mask_prob, nan=0.0)

    # Run post-processing
    instances = _threshold_and_label(mask_prob, threshold, min_area_px, min_hole_area_px)

    # Build output: mask_probability + binary_mask + instances (3 bands)
    binary_f32 = (mask_prob > threshold).astype(np.float32)[np.newaxis, :, :]  # (1, H, W)
    instances_f32 = instances.astype(np.float32)[np.newaxis, :, :]             # (1, H, W)
    mask_prob_out = mask_prob[np.newaxis, :, :]                                 # (1, H, W)
    result = np.concatenate([mask_prob_out, binary_f32, instances_f32], axis=0) # (3, H, W)

    coords = {}
    if y_dim in cube.coords:
        coords[y_dim] = cube.coords[y_dim]
    if x_dim in cube.coords:
        coords[x_dim] = cube.coords[x_dim]

    logger.info("=== Delineate Post-processing UDF END === output shape: %s", result.shape)
    return xr.DataArray(
        result,
        dims=("bands", y_dim, x_dim),
        coords=coords,
    )
