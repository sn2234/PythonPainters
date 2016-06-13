import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import Image

trainDescFile = "..\\Data\\train_info.csv"
trainDataFolder = "..\\Data\\Train"

csvRaw = pd.read_csv(trainDescFile)
trainFiles = {f for f in listdir(trainDataFolder) if isfile(join(trainDataFolder, f))}

# Uncomment for stable experiments
#np.random.seed(123)

csvExistingFiles = csvRaw[[x in trainFiles for x in csvRaw['filename']]]
csvExistingFiles['id'] = np.array(range(len(csvExistingFiles)), dtype=int)
l = len(csvExistingFiles)
# Drop reminder so we could split sequince into pairs
existingIndices = np.array(range(l-(l%2)), dtype=int)
np.random.shuffle(existingIndices)

# Prepare data frame:
#   First image ID
#   First image file name
#   Second image ID
#   Second image file name
#   Same painter flag
trainFrame = pd.DataFrame()

# First add random pairs
halfElems = np.arange(len(existingIndices)/2, dtype=int)
trainFrame['FirstId'] = existingIndices[halfElems*2]
trainFrame['SecondId'] = existingIndices[halfElems*2 + 1]

trainFrame['FirstName'] = csvExistingFiles['filename'].values[existingIndices[halfElems*2]]
trainFrame['SecondName'] = csvExistingFiles['filename'].values[existingIndices[halfElems*2 + 1]]

trainFrame['Same'] = csvExistingFiles['artist'].values[existingIndices[halfElems*2]] == \
                        csvExistingFiles['artist'].values[existingIndices[halfElems*2 + 1]]

# Now we need to add more true entries as in random dataset we'll get about 0.002 such entries
# Let's add one entry for each artist
groups = csvExistingFiles.groupby('artist')

#tt.append({'A':12, 'B':'x'}, ignore_index=True)
#tt.loc[len(tt)] = [12,'x']

for name, group in groups:
   if len(group) > 1:
       [idFirst, idSecond] = np.random.permutation(len(group))[:2]
       trainFrame.loc[len(trainFrame)] = {
           'FirstId' : group['id'].values[idFirst],
           'SecondId' : group['id'].values[idSecond],
           'FirstName' : group['filename'].values[idFirst],
           'SecondName' : group['filename'].values[idSecond],
           'Same' : True
           }

# Experiment #1
# Check corellation between colors histogram and painter.
# 1. Extract image and artist from csvExistingFiles
subFrame = csvExistingFiles.sample(100)

dataFrame = pd.DataFrame()
dataFrame['id'] = subFrame['id']
dataFrame['artist'] = subFrame['artist']

# 2. Load and transform image into histogram
def processImage(x):
    print("Processing file: {0}".format(x))
    return preprocessing.minmax_scale(Image.calcImageHistFast(join(trainDataFolder, x), 50).astype(float))

dataFrame['hist'] = subFrame['filename'].map(processImage)

# 3. Build kNN database {histogram => artist}
knn = KNeighborsClassifier()
knn.fit(np.vstack(dataFrame['hist'].values), dataFrame['artist'].values)

# 4. Check it against trainFrame
checkFrame = trainFrame.sample(100)

xx = knn.predict(np.vstack(checkFrame['FirstName'].map(processImage))) == \
     knn.predict(np.vstack(checkFrame['SecondName'].map(processImage)))

accuracy_score(checkFrame['Same'], xx)
# In more advanced scenario, split csvExistingFiles into two parts, one for kNN and second for validation
