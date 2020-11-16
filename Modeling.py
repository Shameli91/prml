import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
import keras
from skimage.transform import rescale, resize, downscale_local_mean

path = './data/preprocessed/'
datapartitions = ['ff1010bird_' + str(i+1) for i in range(10)] + ['warblrb10k_public_' + str(i+1) for i in range(10)]
preprocessors = ['_specs', '_mfcc', '_joint']
data = []
labels = []
start = time.time()
for partition in datapartitions:
    data.append(np.load(path + partition + preprocessors[2] + '.npy'))
    labels.append(np.load(path + partition + '_labels.npy'))
    print(f'{partition} ready')

data = np.concatenate(data)
labels = np.concatenate(labels)


model = Sequential([
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(32, (5,5), input_shape=(32,650, 6), activation='relu'),
    keras.layers.Conv2D(32, (5,5), activation='relu'),
    keras.layers.SpatialDropout2D(0.1),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64, (5,5), activation='relu'),
    keras.layers.SpatialDropout2D(0.1),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.SpatialDropout2D(0.1),
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),

    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=10000,
    decay_rate=0.97
)

Callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.0001,
        patience=10,
        mode='auto',
        restore_best_weights=True
        )
]

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(data, labels, batch_size = 256, epochs=50, verbose=True, validation_split = 0.1)

model.save('./models/samuli3')

print("Execution time %s minutes" % ((time.time() - start_time)/60))