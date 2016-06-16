
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram
from sklearn import preprocessing
from pyemd import emd

#img = io.imread("..\\Data\\Train\\10.jpg")

#imgComposed = np.zeros((img.shape[0], img.shape[1]), dtype=int32)

#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        imgComposed[i, j] = 0x10000*img[i,j,0] + 0x100*img[i,j,1] + 0x10000*img[i,j,2]

#imgNorm = preprocessing.minmax_scale(imgComposed.astype(float))

#np.histogram(imgNorm)

def convColors(a):
    return 0x10000*a[0] + 0x100*a[1] + 0x10000*a[2]

def calcImageHist(imagePath, nbins):
    img = io.imread(imagePath)

    imgComposed = np.apply_along_axis(convColors, 2, img)

    imgNorm = preprocessing.minmax_scale(imgComposed.astype(float))[0]

    return np.histogram(imgNorm, bins=nbins)

def calcImageHistFast(imagePath, nbins):
    img = io.imread(imagePath, as_grey=True)

    return np.histogram(preprocessing.minmax_scale(img), bins=nbins)[0]

def chiSquareDistance(h1, h2):
    return np.sum((h1-h2)**2/(h1+h2+1e-10))

def emdDistance(h1, h2):
    return emd(h1, h2, np.ones((h1.shape[0], h1.shape[0]))/2)

