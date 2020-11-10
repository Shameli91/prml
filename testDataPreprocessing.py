import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import time



windowms = 64
hop_size = 1/4
freq_bins = 32
specs = []
mfcc = []
joint = []
i = 1
pad_to = 650
path = './main/testDataAudio/'
dataset_name = 'testData'
start = time.time()

def make_uniform(array, lenght):
    if array.shape[1] < lenght:
        missing = lenght - array.shape[1]
        new_array = np.pad(array, ((0, 0), (0, missing)), 'constant', constant_values=(0,0))
    elif array.shape[1] > lenght:
        new_array = array[: , :lenght]
    elif array.shape[1] == lenght:
        new_array = array
    return new_array
            

for i in range(0, 4512):
    audio, rate = librosa.load(path + str(i) + '.wav')
    hop = int(np.round(rate * windowms / (2000) * hop_size)*2)  

    melspectogram = librosa.feature.melspectrogram(audio, sr= rate, n_fft= hop * 8, hop_length= hop, power = 4, n_mels=freq_bins)
    melSpec_dB = librosa.power_to_db(melspectogram, ref=np.max)
    melSpec_dB = make_uniform(melSpec_dB, lenght=pad_to)
        
    cepstogram = librosa.feature.mfcc(audio, sr = rate, n_mfcc = freq_bins, hop_length= hop)
    cepstogram = make_uniform(cepstogram, lenght=pad_to)

    specs.append(melSpec_dB)
    mfcc.append(cepstogram)
    joint.append(np.array([melSpec_dB, cepstogram]).reshape([freq_bins, pad_to, 2]))
    print(f'Sound {i} preprocessed')
    i = i + 1
    
np.save('./main/data/preprocessed/' + dataset_name + '_specs', np.array(specs), allow_pickle=True)
np.save('./main/data/preprocessed/' + dataset_name + '_mfcc', np.array(mfcc), allow_pickle=True)
np.save('./main/data/preprocessed/' + dataset_name + '_joint', np.array(joint), allow_pickle=True)
print(f'Execution time for {dataset_name} preprocessing {time.time() - start} seconds')
print(f'{dataset_name} saved to ./main/data/preprocessed/')