import torch
import torchaudio
import matplotlib.pyplot as plt

filename = r"E:\datas\speech_commands_v0.01\bird\00f0204f_nohash_0.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
specgram = torchaudio.transforms.Spectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
# specgram = torchaudio.transforms.MelSpectrogram()(waveform)
#
# print("Shape of spectrogram: {}".format(specgram.size()))


plt.show()