import numpy as np
import time as time
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
import csv
import os

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(
    physical_devices[0], enable=True
)

model01 = keras.models.load_model('./models/model01/')

with open('./main/predictions/predictions01.csv', mode='w', newline='') as file:
    answerWriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    answerWriter.writerow(['ID', 'Predicted'])
    for i in range(0, 4512):
        print(i)
        img = cv2.imread('./main/testDataImages/' + str(i) + '.wav.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=0)
        prediction = model01.predict(gray)
        if (prediction[0][1] > 0.5):
            answerWriter.writerow([str(i), str(1)])
        else:
            answerWriter.writerow([str(i), str(0)])

