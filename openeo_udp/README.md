# openEO UDP pipeline for Delineate-Anything

This folder contains the full openEO implementation for Delineate-Anything:

- process graph builders
- inference/post-processing UDFs
- UDP generation/registration script
- test scripts (remote and local)

## Folder structure

- `process_graph/delineate_onnx.py`  
  Main process graph builders and default job options.
- `process_graph/generate_udp.py`  
  Builds UDP JSON and optionally registers the UDP on backend.
- `process_graph/delineate_anything_udp.json`  
  Generated UDP definition (written by `generate_udp.py`).
- `udf/delineate_inference.py`  
  ONNX inference UDF (expects dependency archives with `onnxruntime` and model).
- `udf/delineate_postprocess.py`  
  Post-processing UDF (thresholding + connected components).
- `tests/test_udp.py`  
  End-to-end test loading the generated UDP JSON.
- `tests/test_local_udf.py`  
  Local UDF debugging with NetCDF input.

## Prerequisites

1. openEO account on CDSE backend.
2. Access to dependency archives:
   - ONNX runtime archive (`onnx_deps_python311.zip#onnx_deps`)
   - model archive (`DelineateAnything.zip#DelineateAnything`)
3. Python environment with project dependencies installed.

## Generate and register the UDP

Run:

```bash
python openeo_udp/process_graph/generate_udp.py
```

This will:

1. build the process graph,
2. write `process_graph/delineate_anything_udp.json`,
3. register/update UDP `delineate_anything` (if `REGISTER=True`).

## Run remote UDP test

Run:

```bash
python openeo_udp/tests/test_udp.py
```

The script:

1. loads `process_graph/delineate_anything_udp.json`,
2. submits a batch job with `DEFAULT_JOB_OPTIONS`,
3. downloads job results to `openeo_udp/tests/test_outputs/`.

## Exposed UDP parameters

The UDP exposes:

- `spatial_extent` (GeoJSON Polygon)
- `temporal_extent` (`[start, end]`)
- `processing_options` (object)
- `max_cloud_cover`

`processing_options` supports:

- `confidence_threshold`
- `mask_threshold`
- `min_area_px`
- `min_hole_area_px`

Defaults:

- `confidence_threshold = 0.15`
- `mask_threshold = 0.2`
- `min_area_px = 10`
- `min_hole_area_px = 10`

## Geometry input format

The UDP parameter is named `spatial_extent`, but it must be a **GeoJSON geometry** (Polygon), for example:

```json
{
  "type": "Polygon",
  "coordinates": [[[4.95, 50.95], [5.13, 50.95], [5.13, 51.13], [4.95, 51.13], [4.95, 50.95]]]
}
```

## Notes

- If you see `ModuleNotFoundError: onnxruntime` in backend logs, verify job options include `udf-dependency-archives`.
- Keep process graph and UDF changes in sync when adjusting thresholds or output band schema.
