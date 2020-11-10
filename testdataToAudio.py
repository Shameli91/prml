import numpy as np
from scipy.io.wavfile import write

for i in range(0, 4512):
    data = np.load('./bird-audio-detection/' + str(i) + '.npy')
    write('./main/testData/' + str(i) + '.wav', 48000, data)