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
      num_workers: [4, 4]           # Number of postprocessing and polygonization workers (X and Y axes). 
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