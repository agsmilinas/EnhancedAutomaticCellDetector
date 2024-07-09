import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

def plot_high_res_png(im, pts, output):
    print("plotting high res .png file...", flush=True)

    f = plt.figure(figsize=(10,10))
    plt.imshow(im, cmap='gray')
    plt.plot(pts[:, 1], pts[:, 0], 'w.', markersize=0.05)
    plt.tight_layout()
    plt.show()
    print("saving plot ...")
    f.savefig(output, dpi=3500)  # Adjust the dpi value as needed
    print("finished ...", flush=True)

def cv2_plot_high_res_png(im, pts, output):
    print("cv2 high res .png file...", flush=True)
    
    # Convert image to BGR format if it is grayscale
    if len(im.shape) == 2:
        im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        im_color = im.copy()
    
    # Overlay the points on the image with very small red circles
    for pt in pts:
        cv2.circle(im_color, (int(pt[1]), int(pt[0])), radius=1, color=(0, 0, 255), thickness=-1)

    # Save the image
    print("saving plot ...")
    cv2.imwrite(output, im_color)
    print("finished ...", flush=True)

if __name__ == "__main__":
    print("reading 0.5 micron BB HIPP data...", flush=True)
    Image.MAX_IMAGE_PIXELS = None
    image_path = os.getenv('PNG_UPSAMPLED_PATH')
    pts_path = os.getenv('PTS_ANISOF_NPY_PATH')
    res_path = os.getenv('RES_PTS_ANISOF_PNG_PATH')
    res_cv2_path = os.getenv('CV2_RES_PTS_ANISOF_PNG_PATH')

    print("reading BigBrain 0.5 micron hippocampus image data ...", flush=True)
    im = Image.open(image_path).convert('L')
    im = np.array(im)
    print("reading detected POINTS data ...", flush=True)
    pts = np.load(pts_path)
    print(len(pts), flush=True)
    plot_high_res_png(im, pts, res_path)
    cv2_plot_high_res_png(im, pts, res_cv2_path)




