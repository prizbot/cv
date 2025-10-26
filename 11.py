import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_map_sgbm(left_image_path, right_image_path):
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    if left_image is None or right_image is None:
        print("Error: Could not load one or both images")
        return None

    if left_image.shape != right_image.shape:
        right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))

    min_disp, num_disp, block_size = 0, 16*6, 7
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8*3*block_size**2,
        P2=32*3*block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    plt.figure(figsize=(12, 5))
    plt.subplot(1,3,1); plt.title("Left Image"); plt.imshow(left_image, cmap='gray'); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Right Image"); plt.imshow(right_image, cmap='gray'); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Disparity Map"); plt.imshow(disparity_normalized, cmap='plasma'); plt.axis("off")
    plt.show()

    return disparity_normalized

disparity_map = compute_disparity_map_sgbm(r"D:\dl\ryt.png", r"D:\dl\sstero.png")
