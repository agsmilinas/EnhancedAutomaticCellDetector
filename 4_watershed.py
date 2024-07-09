import numpy as np
import cv2
import os
from PIL import Image
import time


if __name__ == "__main__":

    Image.MAX_IMAGE_PIXELS = None


    start_time = time.time()

    # Step 1: Loading and Normalizing Image
    image_path = os.getenv('PNG_UPSAMPLED_PATH')
    pts_path = os.getenv('PTS_ANISOF_NPY_PATH')
    output_path = os.getenv('WATERSHED_IMG_PATH')
    output_path_binary = os.getenv('BINARY_WATERSHED_IMG_PATH')

    print(f"Loading image from {image_path}")
    print(f'image path: {image_path}')
    print(f'pts_path: {pts_path}')
    print(f'output_path: {output_path}')

    im = Image.open(image_path).convert('L')
    image = np.array(im)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")


    if image.dtype == np.float64:
        print("Normalizing image ...")
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.astype(np.uint8)

    print(f" image shape: {image.shape}, dtype: {image.dtype}")

    # Convert grayscale image to 3-channel image
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print(f"Converted image shape: {image_color.shape}, dtype: {image_color.dtype}")

    # Step 2: Loading and Checking Centroids
    print(f"Loading centroids from {pts_path}")
    centroids = np.load(pts_path)

    print(f"Centroids shape: {centroids.shape}, dtype: {centroids.dtype}")

    # Step 3: Applying Watershed
    markers = np.zeros(image.shape[:2], dtype=np.int32)
    for i, (x, y) in enumerate(centroids):
        markers[int(x), int(y)] = i + 1

    print("Applying watershed algorithm ...")
    cv2.watershed(image_color, markers)

    print("Watershed algorithm applied successfully.")

    # Step 4: Saving Output
    # Convert markers to an image for visualization (e.g., normalizing values for display)
    markers_image = np.zeros_like(image, dtype=np.uint8)
    markers_image[markers == -1] = 255  # Mark the boundaries


    # Step 5: Creating Binary Segmentation
    # Create a binary mask where the foreground is marked as 255 (white) and the background as 0 (black)
    binary_segmentation = np.zeros_like(image, dtype=np.uint8)
    binary_segmentation[markers > 0] = 255  # Mark all regions with markers greater than 0 as foreground


    print(f"Saving output to {output_path}")
    cv2.imwrite(output_path, markers_image)
    print("Output saved successfully.")

    print(f"Saving binary segmentation to {output_path_binary}")
    cv2.imwrite(output_path_binary, binary_segmentation)
    print("Binary segmentation saved successfully.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Watershed segmentation Execution time: {total_time}", flush=True)