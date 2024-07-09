import cupy as cp
import numpy as np
import time
import math
import concurrent.futures
from cucim.skimage import morphology as cu_morphology, feature as cu_feature, filters as cu_filters
import os


# Initialize GPUs
num_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Number of GPUs: {num_gpus}", flush=True)

def process_gpu_batch(batch, gpu_id, min_neuron, global_threshold):
    with cp.cuda.Device(gpu_id):
        try:
            imaniso_gpu = cp.asarray(batch)
            mask = imaniso_gpu < global_threshold
            mask = cu_morphology.remove_small_objects(mask, min_size=min_neuron, connectivity=1)
            coordinates = cu_feature.peak_local_max(imaniso_gpu.max() - imaniso_gpu, min_distance=int(math.ceil(math.sqrt(min_neuron/3.14))))
            mask_np = cp.asnumpy(mask)
            coordinates_np = cp.asnumpy(coordinates)
            valid_coords = [coord for coord in coordinates_np if coord[0] < batch.shape[0] and coord[1] < batch.shape[1]]
            return [tuple(coord) for coord in valid_coords if mask_np[tuple(coord)]]
        except Exception as e:
            print(f"Error on GPU {gpu_id}: {e}", flush=True)
            return []

def process_in_batches(imaniso, min_neuron, otsu_corr, batch_size, overlap=20):
    height, width = imaniso.shape
    all_coordinates = set()
    imaniso_gpu = cp.asarray(imaniso)
    global_threshold = cu_filters.threshold_otsu(imaniso_gpu) * otsu_corr
    cp.cuda.Stream.null.synchronize()
    overlap_size = int(batch_size * (overlap / 100))

    # Prepare batches for processing
    batches = []
    for y in range(0, height - overlap_size, batch_size - overlap_size):
        for x in range(0, width - overlap_size, batch_size - overlap_size):
            batch = imaniso[y:y+batch_size, x:x+batch_size]
            batches.append((batch, y, x))

    # Process batches in parallel on GPUs
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(process_gpu_batch, batch, gpu_id % num_gpus, min_neuron, global_threshold) for gpu_id, (batch, y, x) in enumerate(batches)]
        for future, (_, y, x) in zip(futures, batches):
            coords = future.result()
            adjusted_coords = {(cy + y, cx + x) for cy, cx in coords}
            all_coordinates.update(adjusted_coords)

    return list(all_coordinates)

# Main processing

if __name__ == "__main__":
    print("reading 0.5 micron BB HIPP data...",flush = True)
    im_aniso_path = os.getenv('ANISOF_NPY_PATH')
    output = os.getenv('PTS_ANISOF_NPY_PATH')
    im_aniso = np.load(im_aniso_path)

    print("running process...",flush = True)
    start_time = time.time()
    pts = process_in_batches(im_aniso, min_neuron=50, otsu_corr=0.85, batch_size=256)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"process_in_batches GPU Execution time: {total_time}", flush=True)
    print("finished process...",flush = True)

    if pts:
        print(len(pts),flush = True)
        np.save(output, pts)
    else:
        print("No points detected due to errors.",flush = True)
