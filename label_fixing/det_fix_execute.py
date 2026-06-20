import os
import concurrent.futures
from tqdm import tqdm
#from osgeo import gdal
from scipy import ndimage

from helpers import save_to
from detect_and_fix_anomalies import *

# --- Configuration ---
ROOT = r"FBIS-v2"
ROOT_IMAGES = os.path.join(ROOT, "images")
ROOT_LABELS = os.path.join(ROOT, "labels")
DST_IMAGES = os.path.join(ROOT, "images_modified")
SATELLITES = ["_S2_", '_PL_']


NUM_WORKERS = os.cpu_count() // 2
CHUNK_SIZE = 16

# --- Worker Function ---
def process_single_image(task_args):
    """
    Worker target executing the full processing pipeline for a single image.
    Accepts a single tuple to facilitate clean pool mapping.
    """
    name, src_img_path, src_label_path, dst_img_path = task_args
    
    try:
        # 1. Load data
        img = load_image(src_img_path)
        label = load_label(src_label_path, img)

        # 2. Heavy processing pipeline
        if '_S2_' in name:
            _, _, dev_labels_map = find_deviating_labels_dynamic(
                img, label, [0.1, 0.2, 0.5], [0.15, 0.1, 0.05], [256, 96, 32], 
                (32, 1.5, 128, 1.0), (13, 13), 3
            )
            dampen = synthesize_low_quality_regions(img, label, dev_labels_map, 15, bounds=[32, 32, 32])

            edge_strength = get_edge_strength(dampen, label)

            edge_instances_mask = create_khalimsky_grid(label)
            edge_instances, _ = ndimage.label((edge_instances_mask <= 2) & (edge_instances_mask > 0))

            edge_instances[::2, ::2] = 0
            edge_mask, _ = detect_problematic_edges(edge_strength, edge_instances, 8, 16, 0.75)
            burned = create_image_res_via_khalimsky_downsample(dampen, 0.5 * edge_mask * (edge_strength < 24))
        else:
            _, _, dev_labels_map = find_deviating_labels_dynamic(
                img, label, [0.25, 0.35, 0.5], [0.15, 0.1, 0.05], [512, 192, 64], (32, 1.5, 128, 1.0), (13, 13), 3
            )
            dampen = synthesize_low_quality_regions(img, label, dev_labels_map, 15, bounds=[48, 48, 48])

            edge_strength = get_edge_strength(dampen, label)

            edge_instances_mask = create_khalimsky_grid(label)
            edge_instances, _ = ndimage.label((edge_instances_mask <= 2) & (edge_instances_mask > 0))

            edge_instances[::2, ::2] = 0
            edge_mask, _ = detect_problematic_edges(edge_strength, edge_instances, 8, 16, 0.7)
            burned = create_image_res_via_khalimsky_downsample(dampen, 0.3 * edge_mask * (edge_strength < 16))

        # 3. Save output (inherits geotransform/projection from original image via your save_to helper)
        save_to(burned, dst_img_path, src_img_path, 3, False, True)
        return True, name
        
    except Exception as e:
        # Fail gracefully so one corrupted image or IO issue doesn't crash the whole run
        return False, f"Error processing {name}: {str(e)}"


# --- Task Generator ---
def generate_tasks(images_names):
    """Generator to yield task arguments one by one, keeping memory usage minimal."""
    for name in images_names:
        if not any(sat in name for sat in SATELLITES):
            continue

        # Check for .tif to ensure alignment, modify extension dynamically if needed
        label_name = name.replace('.tif', '.txt')
        
        src_img_path = os.path.join(ROOT_IMAGES, name)
        src_label_path = os.path.join(ROOT_LABELS, label_name)
        dst_img_path = os.path.join(DST_IMAGES, name)
        
        yield (name, src_img_path, src_label_path, dst_img_path)


# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(DST_IMAGES, exist_ok=True)

    print("Scanning input directory...")
    images_names = [f for f in os.listdir(ROOT_IMAGES) if f.lower().endswith(('.tif', '.tiff'))]
    total_files = len(images_names)
    print(f"Found {total_files} images to process.")

    # Dynamically scale workers. Adjust max_workers if disk IO bound vs CPU bound.
    num_workers = np.maximum(1, NUM_WORKERS)
    print(f"Launching pool with {num_workers} workers...")

    # Initialize error tracking log
    failed_tasks = []

    # Using ProcessPoolExecutor for heavy CPU workloads (bypasses the GIL)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        # Instantiate task generator
        tasks = generate_tasks(images_names)
        
        # chunksize prevents worker starvation and overhead. 
        # A value of 4-16 is optimal for tasks taking >1-2 seconds each.
        results = executor.map(process_single_image, tasks, chunksize=CHUNK_SIZE)
        
        # Track progress dynamically
        with tqdm(total=total_files, desc="Processing Images", unit="img") as pbar:
            for success, meta in results:
                if not success:
                    failed_tasks.append(meta)
                pbar.update(1)

    # Error Reporting
    if failed_tasks:
        print(f"\n[WARNING] Completed with {len(failed_tasks)} errors:")
        for error in failed_tasks[:10]:  # Show first 10 errors
            print(f"  -> {error}")
        if len(failed_tasks) > 10:
            print(f"  -> ... and {len(failed_tasks) - 10} more errors.")
    else:
        print("\nAll tasks completed successfully!")