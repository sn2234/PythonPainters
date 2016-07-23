import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import DataModel
import TestVggKeras

def processImage(x):
    print("Processing file: {0}".format(x))
    return TestVggKeras.extractImageStyle(join(DataModel.trainDataFolder, x))

# Experiment #1
# Check corellation between colors histogram and painter.
# 1. Extract image and artist from csvExistingFiles

numSamples = 5

checkFrame =  DataModel.trainFrame.sample(numSamples)
firstProc = np.vstack(checkFrame['FirstName'].map(processImage))
secondProc = np.vstack(checkFrame['SecondName'].map(processImage))

subFrame = DataModel.csvExistingFiles.sample(numSamples)

dataFrame = pd.DataFrame()
dataFrame['id'] = subFrame['id']
dataFrame['artist'] = subFrame['artist']

# 2. Load and transform image into histogram

dataFrame['hist'] = subFrame['filename'].map(processImage)

# 3. Build kNN database {histogram => artist}
knn = KNeighborsClassifier(n_neighbors=2, metric=TestVggKeras.diffImagesStyles)
knn.fit(np.vstack(dataFrame['hist'].values), dataFrame['artist'].values)

# 4. Check it against trainFrame

xx = knn.predict(firstProc) == knn.predict(secondProc)

print(metrics.accuracy_score(checkFrame['Same'], xx))
print(metrics.confusion_matrix(checkFrame['Same'], xx))
print("Classification report\n{0}".format(metrics.classification_report(checkFrame['Same'], xx)))
