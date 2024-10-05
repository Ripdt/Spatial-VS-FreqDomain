import cv2
import numpy as np

from gauss import gauss_filter
from frequency_domain import low_pass_filter

from os import mkdir
from os.path import exists
from shutil import rmtree

directory = './out'
if exists(directory):
    rmtree(directory)

def show_and_save_img(img : np.ndarray, filename : str) -> None:
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(directory + '/' + filename + '.png', img)

mkdir(directory)
    
# =========== init =========== 
img = cv2.imread('res/lena.png', 0)
show_and_save_img(img, filename='no-modification')

# ========= smooth ==========
img_gauss = gauss_filter(img)
for i in range(0):
    img_gauss = gauss_filter(img_gauss)
    
show_and_save_img(img_gauss, filename='gauss')

radius = 60
img_lp_ideal = low_pass_filter(img=img, radius=radius, lpType=0)
img_lp_gauss = low_pass_filter(img=img, radius=radius, lpType=1)

show_and_save_img(img_lp_ideal, filename='lp-ideal')
show_and_save_img(img_lp_gauss, filename='lp-gauss')