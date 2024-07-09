import os
import numpy as np
from PIL import Image
import concurrent.futures
from scipy.ndimage import zoom


def generate_high_res_png(npy_path,highres_npy_path, png_path):
    # Load the .npy file
    data = np.load(npy_path)

    # Define the upsampling factor (e.g., 2x)
    upsampling_factor = 2

    # Upsample the data using cubic interpolation
    upsampled_data = zoom(data, upsampling_factor, order=3)  # order=3 for cubic interpolation

    # Save the upsampled data to a new .npy file
    np.save(highres_npy_path, upsampled_data)

    # Normalize the data to the range [0, 65535] for saving as an image
    upsampled_data_normalized = (65535 * (upsampled_data - np.min(upsampled_data)) / np.ptp(upsampled_data)).astype(np.uint16)

    # Save the upsampled data as a .png file using PIL
    image = Image.fromarray(upsampled_data_normalized)
    image.save(png_path, format='PNG', compress_level=0)  # compress_level=0 for no compression

    print(f"Upsampling completed. The upsampled data is saved to {highres_npy_path} and {png_path}.")


if __name__ == "__main__":
    npy_path = os.getenv('NPY_PATH')
    highres_npy_path = os.getenv('NPY_UPSAMPLED_PATH')
    png_path = os.getenv('PNG_UPSAMPLED_PATH')

    if npy_path and png_path and highres_npy_path:
        generate_high_res_png(npy_path,highres_npy_path, png_path)
    else:
        print("NPY_UPSAMPLED_PATH and PNG_UPSAMPLED_PATH and NPY_PATH environment variables must be set.")

