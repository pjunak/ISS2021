#################################################
#            Projekt ISS 2021 / 22              #
#         Author: Petr Junák - xjunak01         #
#              Soundfile analysis               #
#################################################

# Some of the code is reused from last year project.

# Library imports
ImportError
import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import spectrogram, lfilter, freqz, tf2zpk, find_peaks, buttord, butter, iirnotch

import soundfile as sf

#################################################

#################### 4.1 ####################
# Audio loading
orig_audio, fs = sf.read('../audio/xjunak01.wav')    # Loading up the original file
# Time axis creation
orig_audio_t = np.arange(orig_audio.size) / fs
# printing min and max
print("Maximální hodnota: ",orig_audio.max(), "Minimální hodnota", np.abs(orig_audio).min())
# Graph creation
plt.figure(figsize=(15,3))
plt.plot(orig_audio_t, orig_audio)
plt.gca().set_xlabel('$t[s]$')
plt.title('Graph of original audio')
plt.tight_layout()
plt.savefig('Original_audio_file.pdf') # Creating the graph as PDFs in src directory

#################### 4.2 ####################
# Centralisation
orig_audio -= np.mean(orig_audio)
# Normalisation
orig_audio /= np.abs(orig_audio).max() 
# Segment Creation
seg_start = 0   # Beginning of segment in seconds
seg_len = 1024/fs  # Length of the segment in sec
seg_shift = 512/fs # Length of the segment shift in sec

segArray = [] 
length = int(((orig_audio_t[-1]*fs)/512)-1) # Array length to hold all segments of orig_audio
for i in range(length):     # Fills segArray with segments from orig_audio
    seg_start_sam = int(seg_start * fs)               # start of the segment in samples
    seg_len_sam = int((seg_start + seg_len) * fs)     # end of the segment in samples

    segment = orig_audio[seg_start_sam:seg_len_sam]
    
    seg_start = seg_start + seg_shift
    segArray.append(segment)

chosen_frame = 52               #  52nd frame chosen
seg = segArray[chosen_frame]    #  Isolating the chosen frame
# Time axis creation
seg_t = (np.arange(seg.size) / fs) + (chosen_frame * seg_shift)
# Graph creation
plt.figure(figsize=(15,3))
plt.plot(seg_t, seg)
plt.gca().set_xlabel('$t[s]$')
plt.title('Chosen frame')
plt.tight_layout()
plt.savefig('chosen_frame.pdf')



#################### 4.3 ####################
# Transformation using library function
Frequencies_lib = np.fft.fft(seg)
# Generates DFT matrix for N = 1024
dftmtx = np.fft.fft(np.eye(1024))
frequencies = dftmtx.dot(seg)
# Multiplying the signal with DFT matrix
# Graph creation
f=np.arange(frequencies.size)/frequencies.size*fs
# Frequencies using DFT matrix
plt.figure(figsize=(15,3))
plt.plot(f[:f.size//2+1], np.abs(Frequencies_lib[:Frequencies_lib.size//2+1]), linewidth=5)
plt.plot(f[:f.size//2+1], np.abs(frequencies[:frequencies.size//2+1]), color = 'purple', linewidth=2)
plt.gca().set_xlabel('$f[Hz]$')
plt.gca().set_title('Frequencies created by my func and library func')
plt.tight_layout()
plt.savefig('Frequencies.pdf')

#################### 4.4 ####################
# Creating spectogram for the whole audio
f, t, sgr = spectrogram(orig_audio, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20) 
# Creating spectogram graph
plt.figure(figsize=(9,7))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.title('Spectogram of original audio')
plt.savefig('Spectogram_original.pdf')

#################### 4.5 ####################
# Deteremining the noise frequencies
freq1 = 848       # The lowest noise frequency determined by analyzing the spectogram
freq2 = freq1 * 2
freq3 = freq1 * 3
freq4 = freq1 * 4
#################### 4.6 ####################
# Substracting the noise frequencies
samples = []
for i in range(orig_audio.size):
    samples.append(i*1/fs)
# generating cosinuses from known noise frequencies
cos1 = np.cos(np.array(samples) * 2 * np.pi * freq1)
cos2 = np.cos(np.array(samples) * 2 * np.pi * freq2)
cos3 = np.cos(np.array(samples) * 2 * np.pi * freq3)
cos4 = np.cos(np.array(samples) * 2 * np.pi * freq4)
# Adding all noise cosinuses into one
cos = cos1 + cos2 + cos3 + cos4
# Creating file from the result cosinus
sf.write("../audio/generated.wav", cos, fs)
# Creating spetogram for the generated audio
f, t, sgr = spectrogram(cos, fs, nperseg=1024, noverlap=512)
# transfer to PSD with with 0 prevention
sgr_log = 10 * np.log10(sgr+1e-20) 
# Creating spectogram graph
plt.figure(figsize=(9,7))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.title('Spectogram of generated audio')
plt.savefig('Spectogram_of_generated.pdf')

#################### 4.7 ####################
# Creating filter
Q = 30.0  # Filter quality factor
# Design notch filter for every noise frequency
b, a = iirnotch(freq1, Q, fs)
freq1, h1 = freqz(b, a, 2048)
filter1 = 10 * np.log10(abs(h1+1e-20))
filtered = lfilter(b, a, orig_audio)
b, a = iirnotch(freq2, Q, fs)
freq2, h2 = freqz(b, a, 2048)
filter2 = 10 * np.log10(abs(h2+1e-20))
filtered = lfilter(b, a, filtered)
b, a = iirnotch(freq3, Q, fs)
freq3, h3 = freqz(b, a, 2048)
filter3 = 10 * np.log10(abs(h3+1e-20))
filtered = lfilter(b, a, filtered)
b, a = iirnotch(freq4, Q, fs)
freq4, h4 = freqz(b, a, 2048)
filter4 = 10 * np.log10(abs(h4+1e-20))
w4, H4 = freqz(b, a)
filtered = lfilter(b, a, filtered)
# Impuls response
N_imp = 10
imp = [1, *np.zeros(N_imp-1)]
h = lfilter(b, a, imp)

plt.figure(figsize=(5,3))
plt.stem(np.arange(N_imp), h, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.title('Impulsní odezva $h[n]$')
plt.savefig('Impuls response.pdf')

#################### 4.8 ####################
# Zero points and poles
z, p, k = tf2zpk(b, a)
plt.figure(figsize=(4,3.5))
# unit circle
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')
plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')
plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.title('Nuly a póly')
plt.savefig('zeroes_and_poles.pdf')
#################### 4.9 ####################
# Frequency characteristic
plt.figure(figsize=(9,3))
plt.plot(freq1 / 2 / np.pi * fs, filter1, color='purple')
plt.plot(freq2 / 2 / np.pi * fs, filter2, color='orange')
plt.plot(freq3 / 2 / np.pi * fs, filter3, color='magenta')
plt.plot(freq4 / 2 / np.pi * fs, filter4, color='green')
plt.gca().set_xlabel('$f[Hz]$')
plt.gca().set_ylabel("Amplitude (dB)", color='blue')
plt.gca().set_xlim([0, 8000])
plt.gca().set_ylim([-40, 10])
plt.tight_layout()
plt.title('Frequency characteristic of filters')
plt.savefig('filter_freq_char.pdf')
#################### 4.10 ####################
# Filtration
f, t, sfgr = spectrogram(filtered, fs)
sfgr_log = 10 * np.log10(sfgr+1e-20)
plt.figure(figsize=(9,7))
plt.pcolormesh(t,f,sfgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()
plt.title('Spectogram of filtered signal')
plt.savefig('spectogram_filtered.pdf')
# Creating file from the result cosinus
sf.write("../audio/filtered_audio.wav", filtered, fs)


# Showing all plots
plt.show()