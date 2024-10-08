import numpy as np
import cv2
from sobel import sobel_sharpening
from convolution import conv2d_sharpening
from gauss import gauss_filter

def canny_edge_detection(img: np.ndarray, low_threshold: int, high_threshold: int, magnitude_scale: float = 1.0) -> np.ndarray:
    img_blurred = gauss_filter(img, padding=True)

    # Função sobel_sharpening para calcular a magnitude dos gradientes
    magnitude = sobel_sharpening(img_blurred)

    # Escalar a magnitude para ajustar a sensibilidade
    magnitude = np.clip(magnitude * magnitude_scale, 0, 255)

    # Cálculo dos gradientes para obter os ângulos
    kernel_sobel_vertical = np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
    kernel_sobel_horizontal = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))

    grad_x = conv2d_sharpening(img_blurred, kernel_sobel_horizontal)
    grad_y = conv2d_sharpening(img_blurred, kernel_sobel_vertical)

    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180

    # Non-maximum Suppression
    nms = np.zeros_like(magnitude, dtype=np.uint8)
    img_height, img_width = img.shape
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            region = magnitude[i-1:i+2, j-1:j+2]
            angle_deg = angle[i, j]

            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                neighbors = [region[1, 0], region[1, 2]]
            elif 22.5 <= angle_deg < 67.5:
                neighbors = [region[0, 2], region[2, 0]]
            elif 67.5 <= angle_deg < 112.5:
                neighbors = [region[0, 1], region[2, 1]]
            else:
                neighbors = [region[0, 0], region[2, 2]]

            if magnitude[i, j] >= max(neighbors):
                nms[i, j] = magnitude[i, j]

    # Double Threshold
    strong_edges = (nms > high_threshold).astype(np.uint8)
    weak_edges = ((nms >= low_threshold) & (nms <= high_threshold)).astype(np.uint8)

    # Hysteresis
    edges = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(strong_edges == 1)
    weak_i, weak_j = np.where(weak_edges == 1)

    edges[strong_i, strong_j] = 255

    for i, j in zip(weak_i, weak_j):
        if np.any(strong_edges[i-1:i+2, j-1:j+2]):
            edges[i, j] = 255

    return edges

if __name__ == '__main__':
    img = cv2.imread('res/lena.png', cv2.IMREAD_GRAYSCALE)
    
    edges = canny_edge_detection1(img, low_threshold=50, high_threshold=150)
    
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()