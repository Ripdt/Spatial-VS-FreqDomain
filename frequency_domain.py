import numpy as np
import cv2

IDEAL_TYPE = 0
GAUSS_TYPE = 1

def create_low_pass_filter(shape, center, radius, lpType=GAUSS_TYPE, n=2) -> np.ndarray:
    """
    Cria um filtro passa-baixa com diferentes tipos (ideal ou gaussiano).
    """
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))  # distância ao centro
    lpFilter = np.zeros((rows, cols), np.float32)

    if lpType == 0:  # Ideal low-pass filter
        lpFilter[d <= radius] = 1
    elif lpType == 1:  # Gaussian low-pass filter
        lpFilter = np.exp(-d**2 / (2 * (radius**2)))

    # Retorna o filtro em um formato de dois canais (para imagem complexa)
    lpFilter_matrix = np.zeros((rows, cols, 2), np.float32)
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter

    return lpFilter_matrix

def low_pass_filter(img: np.ndarray, radius=60, lpType=GAUSS_TYPE) -> np.ndarray:
    # Dimensões da imagem
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Transformada de Fourier (DFT)
    image_f32 = np.float32(img)
    dft = cv2.dft(image_f32, flags=cv2.DFT_COMPLEX_OUTPUT) # type: ignore

    # Shift do centro da DFT
    dft_shift = np.fft.fftshift(dft)

    # Criar filtro passa-baixa e aplicar
    mask = create_low_pass_filter(dft_shift.shape[:2], center=(ccol, crow), radius=radius, lpType=lpType)
    
    # Aplicar a máscara de passa-baixa
    fshift = dft_shift * mask

    # Shift inverso e DFT inversa
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalização da imagem resultante
    filtered_img = np.abs(img_back)
    filtered_img -= filtered_img.min()
    filtered_img = (filtered_img * 255) / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img

if __name__ == '__main__':
    # Carrega uma imagem em escala de cinza
    img = cv2.imread('res/lena.png', cv2.IMREAD_GRAYSCALE)
    
    # Aplica o filtro passa-baixa gauss
    filtered_image = low_pass_filter(img, radius=60, lpType=GAUSS_TYPE)
    
    # Exibir imagem filtrada
    cv2.imshow('Imagem Filtrada - GAUSS', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Aplica o filtro passa-baixa ideal
    filtered_image = low_pass_filter(img, radius=60, lpType=IDEAL_TYPE)
    
    # Exibir imagem filtrada
    cv2.imshow('Imagem Filtrada - IDEAL', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
