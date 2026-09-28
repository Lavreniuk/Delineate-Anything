# Delineation Configuration Guide

## `batch_sample.yaml` Parameters

- **`base_config`** – Path to the config file used for delineation.
- **`data_root`** – Path to the folder containing image subfolders. Can be relative or absolute.  
  Each subfolder must contain `.tif` images with the same projection, pixel size, and data type.
- **`output_root`** – Path where the results will be saved.
- **`mask_root`** – Path to the LCLU masks corresponding to folders in `data_root`.
- **`temp_root`** – Path for storing temporary files.
- **`keep_temp`** – Whether to keep temporary files. Keeping this `true` can save time during parameter tuning by avoiding repeated LCLU warping.
- **`include`** – List of specific folders from `data_root` to process.
- **`exclude`** – List of folders from `data_root` to skip.

### Mask overrides

If a mask filename does not match the folder name in `data_root`, you can assign the mask explicitly:

    override:
      - entry: FOLDER_NAME_IN_DATA_ROOT
        mask: ABSOLUTE_PATH_TO_THE_MASK_FILE

## Important `config.yaml` Parameters

These parameters must be properly set for accurate results.

### `data_loader`

- **`bands`** – Make sure the order corresponds to RGB channels in your images. Indexing starts at 1.  
  BGR or other combinations are possible, but RGB generally performs best.
- **`nodata_band`** – If one band uniquely represents nodata, specify it here to improve speed.  
  Otherwise, set to `null`.
- **`nodata_value`** – Use `[nodata_r, nodata_g, nodata_b]` when `nodata_band = null`.  
  Otherwise, specify a single scalar value (not an array).

## RAM-Dependent Parameters

### `execution_planner`

Adjust based on system RAM. For 64Gb of RAM you could set:

    execution_planner:
      region_width: 32768
      region_height: 32768

### `postprocess_limits`

Tune based on CPU cores and RAM.

    postprocess_limits:
      num_workers: [4, 4]           # Number of postprocessing and polygonization workers as [Ny, Nx] (Y and X axes). 
                                    # Product should be LESS than total CPU threads. Like [a, b] where a*b = cpu_count - 4.
      queue_tiles_capacity: 32      # Example for 64 GB RAM. Use halve for 32 GB RAM.
      max_tiles_inflight: 64        # Example for 64 GB RAM. Use halve for 32 GB RAM.


## GPU VRAM-Dependent Parameters

### `passes.batch_size`

Set according to available GPU VRAM.  
Rough estimate: 1 image ≈ <1 GB VRAM.  
Set `batch_size` close to available VRAM in GB.

    passes:
      - batch_size: 16

## LCLU Mask Parameters (`mask_info`)

If you are using LCLU masks, verify the following parameters:

    mask_info:
      range: N+1                   # For integer masks in range [0, N] (including nodata), set to N+1. Else use null.
      filter_classes: [...]       # Classes used to fully remove fields in case of excessive overlap.
      clip_classes: [...]         # Classes to subtract from field polygons.

## Tile seams (`passes[].delineation_config`)

Fields that neighbouring tiles fail to merge end up cut by a straight line on the tile grid (every 1280 m for Sentinel-2).
With `merge_tile_seams: true` (the default) such pieces are joined after inference if, on the tile border:
- they touch for at least `seam_min_contact_px` pixels, covering at least `seam_min_coverage` of the longer cut edge;
- both cut edges start and end at the same place (within `max(seam_end_tolerance_px, seam_end_tolerance_rel * contact)`),
  or the smaller piece is only a strip along the border (at most `seam_max_strip_depth_px` deep);
- the tile centred on that border, which sees it with half a tile of context on both sides, did not see a boundary
  there: along at least `seam_min_evidence` of the contact where it saw one, it saw one field across the whole window
  of `seam_split_window_px` px on both sides of the border;
- the mean colour on both sides differs by at most `seam_max_colour_diff` (0-1 scale);
- each piece is joined with at most one piece on the other side of a given border (its longest contact).

`seam_split_window_px: 1` (the default) checks only the pixels right at the border and removes most straight cuts.
A wider window also catches real boundaries a few px off the border, where the tile grid cut two fields at once:
with `8`, about half as many neighbouring fields get wrongly joined (checked against reference fields in 100
countries), but more straight cuts stay.

Raising the thresholds joins fewer pieces; lowering them removes more cuts but may join neighbouring fields that
happen to meet exactly on a tile border (for example, parallel strip fields). Set `merge_tile_seams: false` to disable.
Requires `tile_step: 0.5` (the default).
