import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft
from scipy import signal
import os
from scipy.io.wavfile import write
from scipy.signal import resample, spectrogram
import os

def read_voice(path):
    
    rate, data = wavfile.read(path)
    Amplitude = rfft(data)
    Frequency = rfftfreq(len(data), 1 / rate)
    
    return Frequency, Amplitude

def plotAll(sampling_rate, data, name):
    freqs = rfftfreq(len(data), 1/sampling_rate)
    time = range(len(data))
    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    # plotting wave based on time
    axs[0].plot(time, data)
    axs[0].set_xlabel('Time(s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title(f"waveform of {name} song")

    fft_data = rfft(data)    # real fast fourier transform
    amplitudes = np.abs(fft_data)
    # plotting fourier transform based on frequency
    axs[1].plot(freqs, amplitudes)
    axs[1].set_xlabel('Frequency(Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title(f'Fourier transform of {name} song')
    # plotting spectogram
    freq, time_spec, spectrogram = signal.spectrogram(data, fs=sampling_rate, nfft=2000, window=signal.windows.hann(2000), scaling='spectrum')

    spec = axs[2].pcolormesh(time_spec, freq, 10*np.log10(spectrogram))
    axs[2].set_xlabel('Time(s)')
    axs[2].set_ylabel('Frequency(Hz)')
    axs[2].set_title(f'Spectrogram of {name} song')
    plt.colorbar(spec, ax=axs[2], label='Power/frequency [dB/Hz]', orientation='vertical')
    plt.savefig(f"{name}_plot.png")
    
def low_pass_filter(Frequency, Amplitude, F, t):
    high_freqs = np.where(Frequency > F + t)
    low_freqs = np.where(Frequency < F - t)
    too_high_amp = np.where(Amplitude > 1e8)
    filtered_amplitude = np.copy(Amplitude)
    filtered_amplitude[high_freqs] = 0
    filtered_amplitude[low_freqs] = 0
    filtered_amplitude[too_high_amp] = 0
    return filtered_amplitude

def write_voice(path, freq, data):
    wavfile.write(path, freq, data.astype(np.int32))

def reverse_voice(data , rate): 
    return np.flip(data)

if not os.path.exists("newaudio"):
    os.mkdir("newaudio")

frequence , amplitude = read_voice("potc.wav")
f1, a1 = wavfile.read("potc.wav")
plotAll(f1, a1, "noisy")

def mix_voices(Datas, Rates):
    rate = max(Rates)
    sampled_data = []
    for i in range(len(Datas)):
        nums = int(len(Datas[i]) / Rates[i] * rate)
        temp = resample(Datas[i], nums)
        sampled_data.append(temp)
    
    
    maximum_length = np.max([len(x) for x in sampled_data])
    
            
    for i in range(len(sampled_data)):
        sampled_data[i] = np.pad(sampled_data[i], (0, maximum_length - len(sampled_data[i])), "constant")
    mix = sum(sampled_data)
    
    return mix , rate

def change_voice_speed(path, speed_factor):
    rate, new_data = wavfile.read(path)
    new_rate = rate * speed_factor
    return new_rate, new_data

rate, _ = wavfile.read("potc.wav")
filtered = low_pass_filter(frequence, amplitude, 4000, 8000)
reconstructed = irfft(filtered * 50000).astype(np.int64)
frequence = frequence.astype(np.int64)
write_voice("newaudio\\cleanpotc.wav", rate, reconstructed)
f2, a2 = wavfile.read("newaudio\\cleanpotc.wav")
plotAll(f2, a2, "clean")

new_rate1 , new_signal1 = change_voice_speed("newaudio\\cleanpotc.wav", 2)
write_voice("newaudio\\fastpotc.wav", new_rate1 , new_signal1)
amp1, Freq1 = read_voice("newaudio\\fastpotc.wav")
f3, a3 = wavfile.read("newaudio\\fastpotc.wav")
plotAll(f3, a3, "fast")

new_rate2, new_signal2 = change_voice_speed("newaudio\\cleanpotc.wav", 0.5)
write_voice("newaudio\\slowpotc.wav", int(new_rate2), new_signal2)
f4, a4 = wavfile.read("newaudio\\slowpotc.wav")
plotAll(f4, a4, "slow")

new_rate3, new_signal3 = wavfile.read("newaudio\\cleanpotc.wav")
new_signal3 = reverse_voice(new_signal3, new_rate3)
write_voice("newaudio\\reverspotc.wav", new_rate3, new_signal3)
f5, a5 = wavfile.read("newaudio\\reverspotc.wav")
plotAll(f5, a5, "revers")

new_signal4, new_rate4= mix_voices([new_signal1, new_signal2, new_signal3, reconstructed], [new_rate1, new_rate2, new_rate3, rate])
write_voice("newaudio\\mixpotc.wav", new_rate4, new_signal4.astype(np.int32))
f6, a6 = wavfile.read("newaudio\\mixpotc.wav")
plotAll(f6, a6, "mix")
