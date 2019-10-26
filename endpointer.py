#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

import sys


if len(sys.argv) < 2:
    sys.argv.append('sample.wav')


duration = 10
if len(sys.argv) >= 3:
    duration = int(sys.argv[2])


tf.enable_eager_execution()


# read the wav file
samplerate, rawdata = scipy.io.wavfile.read(sys.argv[1])

assert samplerate == 16000

# chop the file
rawdata = rawdata[:samplerate * duration]

# handle both mono and stereo files
if len(rawdata.shape) == 2:
    input_buffer_full = rawdata[:,0]
else:
    input_buffer_full = rawdata[:]

# normalize (int16_t => float32)
input_buffer_full = input_buffer_full * 0.000030518

signals = tf.reshape(input_buffer_full.astype(np.float32), [1,-1])	# probably two channels, 0=left,1=right
samples_per_10ms = samplerate//100
samples_per_25ms = samplerate//40

# TODO: remove noise

# TODO: use these values are used to prefilter the signal
fband = [125.,7500.]
channels_ep = 40
stfts = tf.signal.stft(signals, frame_length=samples_per_25ms, frame_step=samples_per_10ms)
spectrograms = tf.abs(stfts)

# Warp the linear scale spectrograms into the mel-scale.
num_spectrogram_bins = stfts.shape[-1].value

# Confirmed freq values
mel_lo = 125
mel_hi = 3800
linear_to_mel_weight_matrix_40 = tf.signal.linear_to_mel_weight_matrix(
    channels_ep,
    num_spectrogram_bins,
    samplerate,
    mel_lo,
    mel_hi,
)
mel_spectrograms_40 = tf.tensordot(spectrograms, linear_to_mel_weight_matrix_40, 1)
mel_spectrograms_40.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix_40.shape[-1:]))

# Compute a stabilized log to get log-magnitude mel-scale spectrograms.
log_mel_spectrograms_40 = tf.log(mel_spectrograms_40 + 1e-6)
print(log_mel_spectrograms_40)

# init model
interpreter = tf.lite.Interpreter(model_path='ep.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
tensor_details = interpreter.get_tensor_details()
output_details = interpreter.get_output_details()

print('inputs:',input_details)
for t in tensor_details: print(t['index'], t['name'], t['shape'])
print('outputs',output_details)


list_a = []
list_b = []
for filterbank_energies_ep in log_mel_spectrograms_40[0]:
    # run the endpointer to decide if we should run the RNN
    input_data_ep = [filterbank_energies_ep]
    interpreter.set_tensor(input_details[0]['index'], input_data_ep)
    interpreter.invoke()
    output_data_ep = interpreter.get_tensor(output_details[0]['index'])

    [[a,b]] = output_data_ep
    list_a.append(a)
    list_b.append(b)

    print(input_data_ep)
    print(output_data_ep)


f, (wav,plota,plotb) = plt.subplots(3, sharex=True)
wav.plot(np.linspace(0, len(input_buffer_full)/float(samplerate), len(input_buffer_full), endpoint=False), input_buffer_full)
plota.plot(np.linspace(0, len(input_buffer_full)/float(samplerate), ((len(input_buffer_full)/float(samplerate))*100) -2), list_a)
plotb.plot(np.linspace(0, len(input_buffer_full)/float(samplerate), ((len(input_buffer_full)/float(samplerate))*100) -2), list_b)
plt.show()
