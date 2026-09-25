# 2026 S2 Tile-Edge Case

This case packages the single source scene and configuration used to reproduce
the 1,280 m tile-edge pattern. `reference/2026_s2_test.gpkg` is the output that
was supplied with the report, for visual comparison.

The test performs real `large_v2` inference. It is opt-in because it requires
the model weights and the normal delineation dependencies; feature counts can
vary by runtime/device, so the reference count is not used as a strict
assertion.

Run it with:

```sh
RUN_S2_INTEGRATION=1 PYTHONPATH=. pytest -q tests/test_batch_s2_testcase.py
```

To retain the generated GeoPackage for inspection, set
`S2_TEST_OUTPUT_ROOT`, for example:

```sh
RUN_S2_INTEGRATION=1 S2_TEST_OUTPUT_ROOT=/tmp/s2-test-output PYTHONPATH=. pytest -q tests/test_batch_s2_testcase.py
```

The bundled batch configuration uses paths relative to this directory when run
directly from here. The pytest test resolves the fixture paths itself and writes
its outputs to pytest's temporary directory by default.