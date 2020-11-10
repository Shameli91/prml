import librosa
import librosa.display
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


for filename in os.listdir('../ff1010bird_wav/'):
    
    y, sr = librosa.load('../ff1010bird_wav/' + filename)
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
    librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=sr, fmin=1000, fmax=10000)
    #plt.colorbar(format='%+1.0f dB')
    '''
    
    samplerate, data = wavfile.read('../ff1010bird_wav/' + filename)
    plt.specgram(data, NFFT=512, Fs=samplerate, cmap='Greys')
    '''

    plt.axis('off')
    plt.savefig('../ff1010bird_MEL_jpg_512/' + filename + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.close('all')
    print(filename)