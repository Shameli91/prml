import numpy as np
import time as time
import matplotlib.pyplot as plt
from matplotlib import image
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

#creates labelarray from the given CSV file
labelArray = []
with open('../ff1010bird_metadata_2018.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            labelArray.append(row[2])
            line_count += 1
labelArray = np.array(labelArray)
labelArray = to_categorical(labelArray)

#sorts the names of the pictures into correct order
def sorted_pics(data):
    nameArray = []
    for name in data:
        number = name.strip('.wav.jpg')
        nameArray.append(number)
    for i in range(0, len(nameArray)):
        nameArray[i] = int(nameArray[i])
    nameArray = sorted(nameArray)
    return nameArray

nameArray = sorted_pics(os.listdir('../picsgs'))

#creates the data array
dataArray = []
for name in nameArray:
    dataArray.append(image.imread('../picsgs/' + str(name) + '.wav.jpg'))
dataArray = np.array(dataArray)

print('shape of labelArray: ', labelArray.shape)
print('shape of dataArray: ', dataArray.shape)

inputs = keras.Input(shape=(369, 496, 1), name='img')

x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(inputs)
#x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.1)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.15)(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)

x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
#x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='elu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.SpatialDropout2D(0.2)(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)
outputs = layers.Dense(2, activation='sigmoid')(x)

myCallbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
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
#, callbacks=([myCallbacks])
history = model.fit(dataArray, labelArray, batch_size = 16, epochs = 100, validation_split = 0.1)
'''
model.save('./models/model05/')
test_scores = model.evaluate(testDataArray, testLabelArray, verbose = 2)
print('Test loss: ', test_scores[0])
print('Test accuracy: ', test_scores[1])

'''