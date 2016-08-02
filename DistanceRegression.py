import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn import cross_validation

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import Image
import DataModel

def processImage(x):
    print("Processing file: {0}".format(x))
    return preprocessing.minmax_scale(Image.calcImageHistFast(join(DataModel.trainDataFolder, x), 50).astype(float))

def chiSquareDistanceRaw(h1, h2):
    return (h1-h2)**2/(h1+h2+1e-10)

# 1. Extract image and artist from csvExistingFiles

subFrame =  DataModel.trainFrame.sample(1000)
subFrame['FirstHist'] = subFrame['FirstName'].map(processImage)
subFrame['SecondHist'] = subFrame['SecondName'].map(processImage)
subFrame['Distance'] = [Image.chiSquareDistance(subFrame['FirstHist'].values[i], subFrame['SecondHist'].values[i])
                                                  for i in range(len(subFrame))]

#x = np.vstack([chiSquareDistanceRaw(subFrame['FirstHist'].values[i], subFrame['SecondHist'].values[i])
#                                                  for i in range(len(subFrame))])

x = np.hstack([np.vstack(subFrame['FirstHist'].values), np.vstack(subFrame['SecondHist'].values)])
x = subFrame['Distance'].values
y = subFrame['Same'].values

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(x, y, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], 1)
x_cv = x_cv.reshape(x_cv.shape[0], 1)

classifier = svm.LinearSVC()
classifier.fit(x_train, y_train)

xx = classifier.predict(x_cv)

print(metrics.accuracy_score(y_cv, xx))
print(metrics.confusion_matrix(y_cv, xx))
print("Classification report\n{0}".format(metrics.classification_report(y_cv, xx)))
