import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats
from scipy.io import wavfile
from em import gmm

# FUNCTION TO COMPUTE THE SPECTROGRAM OF AN AUDIO SAMPLE
def spectrogram(freq, signal ,window_size, shift, dft_point):
    sample_size = int((len(signal) - freq*window_size)/(freq*shift) + 1)
    spec = np.zeros((int(dft_point/2),sample_size),dtype=complex)
    for i in range(sample_size):
        sample = np.fft.fft(np.hamming(400)*signal[int(i*shift*freq):int(i*shift*freq) + int(window_size*freq)], dft_point)
        spec[:,i] = sample[0:int(dft_point/2)]
    spec = np.absolute(spec)
    spec = np.log(spec)
    return spec

path_music = './speech_music_classification/train/music'
path_speech = './speech_music_classification/train/speech'

files_music = os.listdir(path_music)
files_speech = os.listdir(path_speech)

window_size = 0.025
shift = 0.010
dft_point = 64
a = 32
b = 2998
total_samples = len(files_music)
music = np.zeros((b * total_samples, a))
speech = np.zeros((b * total_samples, a))

# THIS BLOCK OF CODE COMPUTES THE SPECTROGRAM OF ALL THE MUSIC AND SPEECH FILES
# AND STORES ALL THE FRAMES OBTAINED IN TWO MATRICES
for i in range(total_samples):
    freq, signal = wavfile.read(path_music + '/' + files_music[i])
    spec = spectrogram(freq, signal, window_size, shift, dft_point)
    spec = spec.T
    music[b * i: b * (i+1), :] = spec

for i in range(total_samples):
    freq, signal = wavfile.read(path_speech + '/' + files_speech[i])
    spec = spectrogram(freq, signal, window_size, shift, dft_point)
    spec = spec.T
    speech[b * i:b * (i+1), :] = spec


itr = 100
mix = 5
cov_type = 'full'

# RUN EM ALGORITHM
music_pis, music_means, music_cov, music_log_likelihood = gmm(music, itr, mix, cov_type)


# THE INPUT HYPER-PARAMETERS FOR EM ALGORITHMS
itr = 75
mix = 5
cov_type = 'full'

#RUN EM ALGORITHM
speech_pis, speech_means, speech_cov, speech_log_likelihood = gmm(speech, itr, mix, cov_type)

path_test = './speech_music_classification/test'
files_test = os.listdir(path_test)
window_size = 0.025
shift = 0.010
dft_point = 64
a = 32
b = 2998
total_samples = len(files_test)
test = np.zeros((b * total_samples, a))

# THIS BLOCK OF CODE COMPUTES THE SPECTROGRAM OF ALL THE TEST FILES
# AND STORES ALL THE FRAMES OBTAINED IN MATRIX
for i in range(total_samples):
    freq, signal = wavfile.read(path_test + '/' + files_test[i])
    spec = spectrogram(freq, signal, window_size, shift, dft_point)
    spec = spec.T
    test[b*i:b*(i+1), :] = spec


# FUNCTION TO FIND THE POSTERIOR PROBABILITY
def posterior_prob(x, means, pis, cov):
    val = 0
    for i in range(pis.shape[0]):
        val += pis[i] * scipy.stats.multivariate_normal(means[i], cov[i], allow_singular=True).pdf(x)
    return val[0]

# FUNCTION TO CHECK WHICH CLASS THE FRAME BELONGS TO
def prediction(posterior_speech, posterior_music):
    if posterior_speech > posterior_music:
        return 'speech'
    else:
        return 'music'


# CONVERT THE SPECTROGRAM MATRIX TO A DATAFRAME 
test = pd.DataFrame(test)

# CALCULATE POSTERIOR PROBABILITY AND PREDICT IT'S CLASS FOR EACH TEST SAMPLE
test['posterior_speech'] = test.apply(lambda row: posterior_prob(row[:32], speech_means, speech_pis, speech_cov), axis=1)
test['posterior_music'] = test.apply(lambda row: posterior_prob(row[:32], music_means, music_pis, music_cov), axis=1)
test['predicted'] = test.apply(lambda row: prediction(row.posterior_speech, row.posterior_music), axis=1)

# HERE THE AUDIO SAMPLE WILL BE CLASSIFIED AS MUSIC IF IT HAS MORE NO. OF MUSIC FRAMES THAN SPEECH FRAMES
y = [1]*(total_samples//2)
y.extend([0]* (total_samples//2))
y_pred = []
for i in range(total_samples):
    pred = list(test['predicted'])[b*i: b*(i+1)]
    sp = pred.count('speech')
    mu = pred.count('music')
    if sp > mu:
        y_pred.append(0)
    else:
        y_pred.append(1)

# CALCULATING ACCURACY
true = 0
for i in range(total_samples):
    if y[i] == y_pred[i]:
        true += 1
acc = true/total_samples
acc = acc * 100
print('Accuracy = ' + str(acc))

