#1.Kısım - Sinyal oluşturma
import matplotlib.pyplot as plot
import numpy as np

frequencies = np.arange(1, 500, 1)

samplingFrequency = 1000

s1 = np.empty([0])
s2 = np.empty([0])

start = 1

stop = samplingFrequency + 1

for frequency in frequencies:
    sub1 = np.arange(start, stop, 1)
    sub2 = np.sin(2 * np.pi * sub1 * frequency / samplingFrequency)

    s1 = np.append(s1, sub1)
    s2 = np.append(s2, sub2)

    start = stop + 1
    stop = start + samplingFrequency

plot.subplot(211)
plot.plot(s1, s2)

plot.subplot(212)
plot.specgram(s2, Fs=samplingFrequency)

plot.show()


#2.Kısım - Dosyadan ses verilerini okuyup sinyal işleme
import pyaudio
import wave
from array import array
import matplotlib.pyplot as plot
import scipy.io.wavfile
import numpy as np

fig, axes = plot.subplots(nrows=1, ncols=2)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
FILE_NAME = "ses.wav"

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                    )
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

stream.stop_stream()
stream.close()
audio.terminate()

wavfile = wave.open(FILE_NAME, 'w')
wavfile.setnchannels(CHANNELS)
wavfile.setsampwidth(audio.get_sample_size(FORMAT))
wavfile.setframerate(RATE)
wavfile.writeframes(b''.join(frames))
wavfile.close()

wavfile = wave.open(FILE_NAME, 'r')
samplingFrequency, signalData = scipy.io.wavfile.read('ses.wav')
plot.subplot(211)
plot.plot(signalData)

plot.subplot(212)
plot.specgram(signalData, Fs=samplingFrequency)
plot.show()

#3.Kısım - Real time mikrofon ses verileriyle sinyal işleme
import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from tkinter import TclError

% matplotlib
tk
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

fig, (ax2, ax3) = plt.subplots(2, figsize=(7, 7))

xf = np.linspace(0, 22050, 512)
line_fft, = ax2.plot(xf, np.random.rand(512), '-', lw=2)

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    data = stream.read(CHUNK)
    data_int = struct.unpack('<' + 'H' * 1024, data)
    yf = fft(data_int)
    line_fft.set_ydata(np.abs(yf[0:512]) / (CHUNK * CHUNK * 10))
    spectrum, freqs, t, im = ax3.specgram(data_int, NFFT=1024, Fs=44100)

    try:
        fig.canvas.draw()
        fig.canvas.flush_events()

    except TclError:
        break