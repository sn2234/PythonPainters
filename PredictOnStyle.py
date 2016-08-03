
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt
from os.path import join

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import Sequential

import DataModel
import TestVggKeras

def processImage(x):
    print("Processing file: {0}".format(x))
    return TestVggKeras.extractImageStyle(join(DataModel.trainDataFolder, x))

numSamples = 10

print("Preparing datasets...")
subFrameTrue =  DataModel.trainFrame[DataModel.trainFrame['Same'] == True].sample(numSamples)
subFrameTrue['FirstStyle'] = subFrameTrue['FirstName'].map(processImage)
subFrameTrue['SecondStyle'] = subFrameTrue['SecondName'].map(processImage)
subFrameTrue['Distance'] = [
    TestVggKeras.diffImagesStyles(subFrameTrue['FirstStyle'].values[i], subFrameTrue['SecondStyle'].values[i])
                     for i in range(len(subFrameTrue))]

subFrameFalse =  DataModel.trainFrame[DataModel.trainFrame['Same'] == False].sample(numSamples)
subFrameFalse['FirstStyle'] = subFrameFalse['FirstName'].map(processImage)
subFrameFalse['SecondStyle'] = subFrameFalse['SecondName'].map(processImage)
subFrameFalse['Distance'] = [
    TestVggKeras.diffImagesStyles(subFrameFalse['FirstStyle'].values[i], subFrameFalse['SecondStyle'].values[i])
                     for i in range(len(subFrameFalse))]

combined = np.hstack((
    np.vstack((np.vstack(subFrameTrue['FirstStyle'].values), np.vstack(subFrameFalse['FirstStyle'].values))),
    np.vstack((np.vstack(subFrameTrue['SecondStyle'].values), np.vstack(subFrameFalse['SecondStyle'].values))),
    np.vstack((subFrameTrue['Distance'].values.reshape(numSamples, 1),
               subFrameFalse['Distance'].values.reshape(numSamples, 1)))))

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(combined,
                                                                   np.hstack((subFrameTrue['Same'].values,
                                                                              subFrameFalse['Same'].values)),
                                                                   test_size=0.3)

print("Initializing model...")

model = Sequential()
model.add(Dense(512, input_shape=(combined.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
print("Fitting model...")
model.fit(x_train, y_train)

pp = model.predict(x_cv)

pp_bool = np.isclose(True, pp)
print(metrics.accuracy_score(y_cv, pp_bool))
print(metrics.confusion_matrix(y_cv, pp_bool))
print("Classification report\n{0}".format(metrics.classification_report(y_cv, pp_bool)))
