import numpy as np
import time as time
import matplotlib.pyplot as plt
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
from keras.utils import to_categorical
import csv
import os

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(
    physical_devices[0], enable=True
)

print(os.getcwd())
#creates labelarray from the given CSV file
dataset = []
labelset = []
with open('./ff1010bird_metadata_2018.csv', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        img = cv2.imread('./ff1010bird_MEL_jpg_512/' + row[0] + '.wav.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dataset.append(gray)
        labelset.append(row[2])
print('first dataset done.')

with open('./warblrb10k_public_metadata_2018.csv', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        img = cv2.imread('./warblrb10k_MEL_jpg_512/' + row[0] + '.wav.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dataset.append(gray)
        labelset.append(row[2])
print('second dataset done.')

combined = list(zip(dataset, labelset))
np.random.shuffle(combined)
dataset[:], labelset[:] = zip(*combined)
print('shuffled.')


Xtest = dataset[int(len(dataset)*0.9):]
Ytest = labelset[int(len(labelset)*0.9):]
Xtrain = dataset[:int(len(dataset)*0.9)]
Ytrain = labelset[:int(len(labelset)*0.9)]
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)
Ytest = to_categorical(Ytest)
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Ytrain = to_categorical(Ytrain)

inputs = keras.Input(shape=(369, 496, 1), name='img')

x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(inputs)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)

x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.1)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)

x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.15)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)

x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.2)(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)
outputs = layers.Dense(2, activation='sigmoid')(x)

myCallbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        mode='auto',
        restore_best_weights=True
        )
]

model = keras.Model(inputs=inputs, outputs=outputs, name='model')
print(model.summary())

model.compile(
    loss = keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(0.001),
    metrics = ['accuracy']
)

history = model.fit(Xtrain, Ytrain, batch_size = 8, epochs = 100, verbose=2, callbacks=([myCallbacks]), validation_split = 0.1)

model.save('./models/model01/')

test_scores = model.evaluate(Xtest, Ytest, verbose = 2)
print('Test loss: ', test_scores[0])
print('Test accuracy: ', test_scores[1])

