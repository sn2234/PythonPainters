
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt
from os.path import join

from keras.layers import Input, Dense
from keras.models import Model

import DataModel
import TestVggKeras

def processImage(x):
    print("Processing file: {0}".format(x))
    return TestVggKeras.extractImageStyle(join(DataModel.trainDataFolder, x))

numSamples = 100

subFrame =  DataModel.trainFrame.sample(numSamples)
subFrame['FirstStyle'] = subFrame['FirstName'].map(processImage)
subFrame['SecondStyle'] = subFrame['SecondName'].map(processImage)
subFrame['Distance'] = [
    TestVggKeras.diffImagesStyles(subFrame['FirstStyle'].values[i], subFrame['SecondStyle'].values[i])
                     for i in range(len(subFrame))]

combined = np.hstack((
    np.vstack(subFrame['FirstStyle'].values),
    np.vstack(subFrame['SecondStyle'].values),
    subFrame['Distance'].values.reshape(numSamples, 1)))

(x_train, x_cv, y_train, y_cv) = cross_validation.train_test_split(combined, subFrame['Same'].values, test_size=0.3)

model_input = Input(shape=(combined.shape[1],))
model_pred = Dense(1, activation='sigmoid')(model_input)
model = Model(input=model_input, output = model_pred)
model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=10)
model.evaluate(x_cv, y_cv)

pp = model.predict(x_cv)

pp_bool = np.isclose(True, pp)
print(metrics.accuracy_score(y_cv, pp_bool))
print(metrics.confusion_matrix(y_cv, pp_bool))
print("Classification report\n{0}".format(metrics.classification_report(y_cv, pp_bool)))
