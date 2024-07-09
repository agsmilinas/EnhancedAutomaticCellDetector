import numpy as np
from scipy import ndimage
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os

def anisodiff_subimage(subim, niter, delta, kappa, masks):
    u = subim.astype('float32')  # Using float32 to save memory
    for r in range(niter):
        convolutions = [ndimage.filters.convolve(u, m) for m in masks]
        PM = sum([np.multiply(np.exp(-np.square(conv/kappa)), conv) for conv in convolutions[:4]])
        PM += (1.0/2.0) * sum([np.multiply(np.exp(-np.square(conv/kappa)), conv) for conv in convolutions[4:]])
        u += delta * PM
    return u

def split_image(im, num_strips):
    strip_height = im.shape[0] // num_strips
    return [im[i * strip_height: (i + 1) * strip_height if (i + 1) < num_strips else im.shape[0], :] for i in range(num_strips)]

def combine_strips(strips):
    return np.vstack(strips)

def anisodiff_parallel(im, niter, delta=1.0/7.0, kappa=15, num_threads=8):
    print("Starting anisodiff_parallel...")

    start_time = time.time()
    # Initialize masks
    masks = [
        np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32), # N
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32), # S
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32), # E
        np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32), # W
        np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32), # NE
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32), # SE
        np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32), # SW
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)  # NW
    ]

    # Split image into strips
    strips = split_image(im, num_threads)

    # Process each strip in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = [executor.submit(anisodiff_subimage, strip, niter, delta, kappa, masks).result() for strip in strips]

    # Combine the processed strips
    combined_image = combine_strips(results)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"anisodiff_parallel Execution time: {total_time} seconds")
    return combined_image

def upsample_image(image_path, output_path, scaling_factor=2):
    print("upsampling_factor",scaling_factor)
    with Image.open(image_path) as img:
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)

        # Resize the image using bicubic interpolation
        upsampled_img = img.resize((new_width, new_height), Image.BICUBIC)

        # Save the upsampled image
        upsampled_img.save(output_path)


if __name__ == "__main__":
    print("running 1_parallelized from pythn",flush=True)
    Image.MAX_IMAGE_PIXELS = None

    #GET ENV VARS
    image_path = os.getenv('PNG_UPSAMPLED_PATH')
    output_path = os.getenv('ANISOF_NPY_PATH')
    anisof_iterations_path = os.getenv('ANISOF_ITERATIONS')

    # Convert to integer
    anisof_iterations_int = int(anisof_iterations_path)

    # Print type and value
    print(type(anisof_iterations_int))
    print(anisof_iterations_int)

    im = Image.open(image_path).convert('L')
    im = np.array(im)

    processed_image = anisodiff_parallel(im, niter=anisof_iterations_int, num_threads=8)
    # Save the processed image
    np.save(output_path, processed_image)
