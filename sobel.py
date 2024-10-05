import numpy as np
import cv2
from convolution import conv2d_sharpening

def sobel_sharpening(img : np.ndarray) -> np.ndarray:
    kernel_sobel_vertical = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    kernel_sobel_horizontal = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))

    img_sobel_1 = conv2d_sharpening(img, kernel_sobel_vertical)
    img_sobel_2 = conv2d_sharpening(img, kernel_sobel_horizontal)
    
    return np.hypot(img_sobel_1, img_sobel_2)

# Somente para debug
def sobel_sharpening_optmized(img : np.ndarray) -> np.ndarray:
    sobel_kernel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_kernel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_kernel_x, sobel_kernel_y)
    return np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude)) # type: ignore
