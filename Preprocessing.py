import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import time

def make_uniform(array, lenght):
    if array.shape[1] < lenght:
        missing = lenght - array.shape[1]
        new_array = np.pad(array, ((0, 0), (0, missing)), 'constant', constant_values=(0,0))
    elif array.shape[1] > lenght:
        new_array = array[lenght,:]
    elif array.shape[1] == lenght:
        new_array = array
    return new_array
            
#ff1010bird_wav dataset
def preprocess_dateset(dataset_name, path, labels, pad_to = 650):
    windowms = 64
    hop_size = 1/4
    freq_bins = 32
    specs = []
    mfcc = []
    joint = []
    i = 1
    start = time.time()
    for filename, _, label in labels:
        
        audio, rate = librosa.load(path + filename + '.wav')
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
        
    np.save('./data/preprocessed/' + dataset_name + '_specs', np.array(specs), allow_pickle=True)
    np.save('./data/preprocessed/' + dataset_name + '_mfcc', np.array(mfcc), allow_pickle=True)
    np.save('./data/preprocessed/' + dataset_name + '_joint', np.array(joint), allow_pickle=True)
    np.save('./data/preprocessed/' + dataset_name + '_labels', ff1010bird_labels[:,2])
    print(f'Execution time for {dataset_name} preprocessing {time.time() - start} seconds')
    print(f'{dataset_name} saved to ./data/preprocessed/')


split_to = 10 # In how many parts should the dataset be handled

ff1010bird_labels = np.loadtxt('./data/ff1010bird_metadata_2018.csv', dtype=str, delimiter=',', skiprows=1)
for i in range(split_to):
    start_idx = int(np.round(ff1010bird_labels.shape[0] / split_to) * i)
    end_idx = int(np.round(ff1010bird_labels.shape[0] / split_to) * (i + 1) - 1)
    preprocess_dateset(dataset_name = 'ff1010bird_' + str(i+1), path = './data/ff1010bird_wav/wav/', labels = ff1010bird_labels[start_idx:end_idx])

warblrb10k_public_labels = np.loadtxt('./data/warblrb10k_public_metadata_2018.csv', dtype=str, delimiter=',', skiprows=1)
for i in range(split_to):
    start_idx = int(np.round(ff1010bird_labels.shape[0] / split_to) * i)
    end_idx = int(np.round(ff1010bird_labels.shape[0] / split_to) * (i + 1) - 1)
    preprocess_dateset(dataset_name = 'warblrb10k_public', path = './data/warblrb10k_public_wav/wav/', labels = warblrb10k_public_labels[start_idx:end_idx])

