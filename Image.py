
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram

img = io.imread("..\\Data\\Train\\10.jpg")

imgComposed = np.zeros((img.shape[0], img.shape[1]), dtype=int32)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        imgComposed[i, j] = 0x10000*img[i,j,0] + 0x100*img[i,j,1] + 0x10000*img[i,j,2]
