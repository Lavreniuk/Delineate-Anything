import numpy as np

from methods.main.PostprocWorker import PostprocWorker


def test_find_edge_mapping_merges_small_fragment_fully_contained_in_large_field():
    """Test that fragment fully contained within a larger field is correctly merged.
    
    A small fragment whose overlap is almost entirely contained within a much
    larger already-written field should be merged, even though the fragment is
    so small relative to the big field that the global IoU stays tiny.

    Without case_0_is_asymmetric_containment, find_edge_mapping requires either
    a high global IoU or an edge-touching (odd) id to merge, which rejects this
    legitimate case and leaves the fragment as a separate polygon.
    """
    # "current" (already written) is entirely one big field, id=200.
    current = np.full((10, 10), 200, dtype="int32")

    # "new" only contains a small 3x3 fragment, id=300, fully inside the big field.
    new = np.zeros((10, 10), dtype="int32")
    new[3:6, 3:6] = 300

    area_dict = {200: 1000, 300: 9}
    dst = {}

    PostprocWorker.find_edge_mapping(
        current,
        new,
        dst,
        area_dict,
        merge_iou=0.05,
        merge_edge_iou=0.3,
        merge_edge_pixels=8,
        merge_relative_area_threshold=0.5,
        merge_asymetric_pixel_area_threshold=4,
        merge_asymetric_relative_area_threshold=0.6,
    )

    assert dst == {300: [200]}
