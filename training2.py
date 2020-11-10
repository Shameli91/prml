import librosa
import librosa.display
import os
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
import keras
from skimage.transform import rescale, resize, downscale_local_mean
from keras.utils import to_categorical

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(
    physical_devices[0], enable=True
)

path = './main/data/preprocessed/'
datapartitions = ['ff1010bird_' + str(i+1) for i in range(10)] + ['warblrb10k_' + str(i+1) for i in range(10)]
preprocessors = ['_specs', '_mfcc', '_joint']
data = []
labels = []
start_time = time.time()
for partition in datapartitions:
    data.append(np.load(path + partition + preprocessors[2] + '.npy'))
    labels.append(np.load(path + partition + '_labels.npy'))
    print(f'{partition} ready')

data = np.concatenate(data)
labels = np.concatenate(labels)
labels = to_categorical(labels)

print(data.shape)


inputs = keras.Input(shape=(32, 650, 2), name='img')

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

history = model.fit(data, labels, batch_size = 64, epochs = 100, verbose=1, callbacks=([myCallbacks]), validation_split = 0.1)

model.save('./models/model02/')
