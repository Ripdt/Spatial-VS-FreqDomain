import cv2
import numpy as np

from gauss import gauss_filter
from low_pass_filter import low_pass_filter
from sobel import sobel_sharpening
from high_pass_filter import high_pass_filter
from canny import canny_edge_detection
from metrics import calculate_mse, calculate_rmse, calculate_psnr

from os import mkdir
from os.path import exists
from shutil import rmtree

from time import time

# Criação do diretório de saída
directory = './out'
if exists(directory):
    rmtree(directory)

mkdir(directory)

def show_and_save_img(img: np.ndarray, filename: str) -> None:
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(directory + '/' + filename + '.png', img)

# =========== Inicialização =========== 
img = cv2.imread('res/lena.png', cv2.IMREAD_GRAYSCALE)
show_and_save_img(img, filename='no-modification')

# ========= Filtro Gaussiano ==========
initial_time = time()
img_gauss = gauss_filter(img)
final_time = time()
gauss_time = final_time - initial_time
show_and_save_img(img_gauss, filename='gauss')

# ========= Filtros Passa-baixa ==========
radius = 60
initial_time = time()
img_lp_ideal = low_pass_filter(img=img, radius=radius, lpType=0)  # Passa-baixa ideal
final_time = time()
lp_ideal_time = final_time - initial_time

initial_time = time()
img_lp_gauss = low_pass_filter(img=img, radius=radius, lpType=1)  # Passa-baixa gaussiano
final_time = time()
lp_gauss_time = final_time - initial_time

show_and_save_img(img_lp_ideal, filename='lp-ideal')
show_and_save_img(img_lp_gauss, filename='lp-gauss')

# ========= Filtro Sobel ==========
initial_time = time()
img_sobel = sobel_sharpening(img)
final_time = time()
sobel_time = final_time - initial_time
show_and_save_img(img_sobel, filename='sobel')

# ========= Filtro Canny ==========
initial_time = time()
img_canny = canny_edge_detection(img, 50, 100)
final_time = time()
canny_time = final_time - initial_time
show_and_save_img(img_canny, filename='canny')

# ========= Filtros Passa-alta ==========
radius = 120
initial_time = time()
img_hp_ideal = high_pass_filter(img=img, radius=radius, lpType=0)  # Passa-alta ideal
final_time = time()
hp_ideal_time = final_time - initial_time

initial_time = time()
img_hp_gauss = high_pass_filter(img=img, radius=radius, lpType=1)  # Passa-alta gaussiano
final_time = time()
hp_gauss_time = final_time - initial_time

show_and_save_img(img_hp_ideal, filename='hp-ideal')
show_and_save_img(img_hp_gauss, filename='hp-gauss')

# ========= Comparações com Canny ==========
# Comparar Sobel com Canny
sobel_canny_psnr = calculate_psnr(img_sobel, img_canny)
sobel_canny_rmse = calculate_rmse(img_sobel, img_canny)
sobel_canny_mse = calculate_mse(img_sobel, img_canny)

# Comparar Passa-alta Ideal com Canny
hp_ideal_canny_psnr = calculate_psnr(img_hp_ideal, img_canny)
hp_ideal_canny_rmse = calculate_rmse(img_hp_ideal, img_canny)
hp_ideal_canny_mse = calculate_mse(img_hp_ideal, img_canny)

# Comparar Passa-alta Gaussiano com Canny
hp_gauss_canny_psnr = calculate_psnr(img_hp_gauss, img_canny)
hp_gauss_canny_rmse = calculate_rmse(img_hp_gauss, img_canny)
hp_gauss_canny_mse = calculate_mse(img_hp_gauss, img_canny)

# ========== Cálculo de Métricas ==========
# Comparar Filtro Espacial de Esmaecimento Gaussiano com os filtros Passa-baixa Ideal e Gaussiano
gauss_lp_ideal_psnr = calculate_psnr(img_gauss, img_lp_ideal)
gauss_lp_ideal_rmse = calculate_rmse(img_gauss, img_lp_ideal)
gauss_lp_ideal_mse = calculate_mse(img_gauss, img_lp_ideal)

gauss_lp_gauss_psnr = calculate_psnr(img_gauss, img_lp_gauss)
gauss_lp_gauss_rmse = calculate_rmse(img_gauss, img_lp_gauss)
gauss_lp_gauss_mse = calculate_mse(img_gauss, img_lp_gauss)

# ========== Exibição das Métricas ==========
print("Comparação: Filtro Espacial de Esmaecimento Gaussiano vs Filtros Passa-baixa")
print(f"Gaussiano vs Passa-baixa Ideal - PSNR: {gauss_lp_ideal_psnr}, RMSE: {gauss_lp_ideal_rmse}, MSE: {gauss_lp_ideal_mse}")
print(f"Gaussiano vs Passa-baixa Gaussiano - PSNR: {gauss_lp_gauss_psnr}, RMSE: {gauss_lp_gauss_rmse}, MSE: {gauss_lp_gauss_mse}")
print("\n")

# Comparação: Sobel com Canny
print("Comparação: Sobel com Canny")
print(f"Sobel vs Canny - PSNR: {sobel_canny_psnr}, RMSE: {sobel_canny_rmse}, MSE: {sobel_canny_mse}")
print("\n")

# Comparação: Passa-alta Ideal com Canny
print("Comparação: Passa-alta Ideal com Canny")
print(f"Passa-alta Ideal vs Canny - PSNR: {hp_ideal_canny_psnr}, RMSE: {hp_ideal_canny_rmse}, MSE: {hp_ideal_canny_mse}")
print("\n")

# Comparação: Passa-alta Gaussiano com Canny
print("Comparação: Passa-alta Gaussiano com Canny")
print(f"Passa-alta Gaussiano vs Canny - PSNR: {hp_gauss_canny_psnr}, RMSE: {hp_gauss_canny_rmse}, MSE: {hp_gauss_canny_mse}")
print("\n")

print("Tempos de execução")
print(f"\tGaussiano: {gauss_time} segundos")
print(f"\tPassa-baixa Ideal: {lp_ideal_time} segundos")
print(f"\tPassa-baixa Gaussiano: {lp_gauss_time} segundos")
print("")
print(f"\tSobel: {sobel_time} segundos")
print(f"\tPassa-alta Ideal: {hp_ideal_time} segundos")
print(f"\tPassa-alta Gaussiano: {hp_gauss_time} segundos")