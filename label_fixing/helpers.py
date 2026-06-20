import rasterio
from rasterio.transform import Affine


def save_to(
    data,
    out_path,
    img_path,
    scale=1,
    khalimsky=False,
    silent=False,
):
    """
    Save numpy array preserving CRS and georeferencing from img_path.

    Parameters
    ----------
    data : np.ndarray
        HxW or HxWxC array.
    out_path : str
        Output GeoTIFF path.
    img_path : str
        Reference GeoTIFF.
    scale : float
        Used only for Khalimsky correction.
    khalimsky : bool
        Whether output is a Khalimsky grid.
    """

    with rasterio.open(img_path) as src:
        transform = src.transform
        profile = src.profile.copy()

        orig_width = src.width
        orig_height = src.height

    if data.ndim == 2:
        new_height, new_width = data.shape
        bands = 1
    else:
        new_height, new_width, bands = data.shape

    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    if khalimsky:
        scale_x = (new_width - 1) / orig_width
        scale_y = (new_height - 1) / orig_height

    new_transform = Affine(
        transform.a / scale_x,
        transform.b,
        transform.c,
        transform.d,
        transform.e / scale_y,
        transform.f,
    )

    if khalimsky:
        new_transform = Affine(
            new_transform.a,
            new_transform.b,
            new_transform.c - 0.5 * scale,
            new_transform.d,
            new_transform.e,
            new_transform.f + 0.5 * scale,
        )

    profile.update(
        driver="GTiff",
        width=new_width,
        height=new_height,
        count=bands,
        dtype=data.dtype,
        transform=new_transform,
        compress="lzw",
        nodata=0,
    )

    with rasterio.open(out_path, "w", **profile) as dst:

        if bands == 1:
            dst.write(data, 1)
        else:
            dst.write(data.transpose(2, 0, 1))

    if not silent:
        print(f"Saved: {out_path}")