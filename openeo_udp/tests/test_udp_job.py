#%%

"""Run the generated Delineate-Anything UDP JSON against an openEO backend.

Loads ``openeo_udp/process_graph/delineate_anything_udp.json`` and submits a
batch job by binding the ``geometry`` and ``temporal_extent`` UDP parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import openeo

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openeo_udp.tests.test_udf_cdse import DEFAULT_JOB_OPTIONS  # noqa: E402

UDP_JSON = _REPO_ROOT / "openeo_udp" / "process_graph" / "delineate_anything_udp.json"
BACKEND = "https://openeo.dataspace.copernicus.eu"


def submit_udp_job(
    connection: openeo.Connection,
    geometry,
    temporal_extent,
    *,
    udp_json_path: Path = UDP_JSON,
    title: str = "Delineate-Anything UDP job",
    job_options: dict | None = None,
):
    cube = connection.datacube_from_json(
        str(udp_json_path),
        parameters={"geometry": geometry, "temporal_extent": temporal_extent},
    )
    return cube.create_job(
        title=title,
        job_options=job_options or DEFAULT_JOB_OPTIONS,
    )


def main() -> None:
    conn = openeo.connect(BACKEND).authenticate_oidc()

    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [4.997886465978867, 51.000003053347534],
            [5.0500127304842835, 50.99909971909893],
            [5.052258312621218, 51.04996441050298],
            [5.000075043015026, 51.05086937572316],
            [4.997886465978867, 51.000003053347534],
        ]],
    }

    job = submit_udp_job(
        conn,
        geometry,
        ["2024-05-01", "2024-08-31"],
    )
    job.start_and_wait()

    print(json.dumps({"job_id": job.job_id, "status": job.status()}, indent=2))


if __name__ == "__main__":
    main()

# %%


connection = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()
job = connection.job("j-26091714120441c7aa6b649d8621fbe3")

results = job.get_results()
files = results.download_files()


import rasterio
numpy as np, matplotlib.pyplot as plt; from matplotlib import cm, colors; p=r'openeo_udp/tests/openEO.tif'; out=r'openeo_udp/tests/openEO_color.png'; arr=rasterio.open(p).read(1).astype('float32'); m=np.isfinite(arr); vals=arr[m]; vmin=float(np.percentile(vals,2)) if vals.size else 0.0; vmax=float(np.percentile(vals,98)) if vals.size else 1.0; vmax=vmax if vmax>vmin else vmin+1.0; norm=colors.Normalize(vmin=vmin,vmax=vmax,clip=True); rgba=cm.turbo(norm(np.where(m,arr,vmin))); rgba[...,3]=np.where(m,1.0,0.0); plt.imsave(out, rgba); print(f'Saved: {out}'); print(f'Range used: vmin={vmin:.6g}, vmax={vmax:.6g}')