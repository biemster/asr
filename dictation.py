#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

tf.enable_eager_execution()

# read the wav file
samplerate,rawdata = scipy.io.wavfile.read('sample.wav')
input_buffer_full = rawdata[:,0]
signals = tf.reshape(input_buffer_full.astype(np.float32), [1,-1])	# probably two channels, 0=left,1=right
samples_per_10ms = samplerate /100
samples_per_25ms = samplerate /40
fband = [125.,7500.]
channels = 80
channels_ep = 40
stfts = tf.signal.stft(signals, frame_length=samples_per_25ms, frame_step=samples_per_10ms)
spectrograms = tf.abs(stfts)

# Warp the linear scale spectrograms into the mel-scale.
num_spectrogram_bins = stfts.shape[-1].value
linear_to_mel_weight_matrix_40 = tf.signal.linear_to_mel_weight_matrix(channels_ep, num_spectrogram_bins, samplerate, fband[0], fband[1])
mel_spectrograms_40 = tf.tensordot(spectrograms, linear_to_mel_weight_matrix_40, 1)
mel_spectrograms_40.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix_40.shape[-1:]))
linear_to_mel_weight_matrix_80 = tf.signal.linear_to_mel_weight_matrix(channels, num_spectrogram_bins, samplerate, fband[0], fband[1])
mel_spectrograms_80 = tf.tensordot(spectrograms, linear_to_mel_weight_matrix_80, 1)
mel_spectrograms_80.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix_80.shape[-1:]))

# Compute a stabilized log to get log-magnitude mel-scale spectrograms.
log_mel_spectrograms_40 = tf.log(mel_spectrograms_40 + 1e-6)
log_mel_spectrograms_80 = tf.log(mel_spectrograms_80 + 1e-6)
print(log_mel_spectrograms_40)


models = ['joint','dec','enc0','enc1','ep']
interpreters = {}
input_details = {}
tensor_details = {}
output_details = {}

# init models
for m in models:
    # Load TFLite model and allocate tensors.
    interpreters[m] = tf.lite.Interpreter(model_path=m+'.tflite')
    interpreters[m].allocate_tensors()
    
    # Get input and output tensors.
    input_details[m] = interpreters[m].get_input_details()
    tensor_details[m] = interpreters[m].get_tensor_details()
    output_details[m] = interpreters[m].get_output_details()

    print(m,':')
    print('inputs:',input_details[m])
    for t in tensor_details[m]: print(t['index'], t['name'], t['shape'])
    print('outputs',output_details[m])


# init the stackers
fft_energies_prev = np.array(np.random.random_sample([1, 80]), dtype=np.float32)
fft_energies_prevprev = fft_energies_prev
output_shape_enc0 = output_details['enc0'][0]['shape']
output_data_enc0_prev = np.array(np.random.random_sample(output_shape_enc0), dtype=np.float32)

# init the loop in the decoder
output_shape_dec = output_details['dec'][0]['shape']
output_data_dec = np.array(np.random.random_sample(output_shape_dec), dtype=np.float32)

# run over frames
list_a = []
list_b = []
list_c = []
for filterbank_energies_ep in log_mel_spectrograms_40[0]:
	# run the endpointer to decide if we should run the RNN
	input_data_ep = [filterbank_energies_ep]
	interpreters['ep'].set_tensor(input_details['ep'][0]['index'], input_data_ep)
	interpreters['ep'].invoke()
	output_data_ep = interpreters['ep'].get_tensor(output_details['ep'][0]['index'])
	print(input_data_ep)
	print(output_data_ep)


	# feed the RNN
	[[a,b]] = output_data_ep
	list_a.append(a)
	list_b.append(b)
	list_c.append(a-(b*10))
	if a > 0 and b > 0:
		# input_data_enc0 = np.array(np.random.random_sample(input_details['enc0'][0]['shape']), dtype=np.float32)
		input_data_enc0_stacked = np.concatenate((fft_energies_prevprev,fft_energies_prev,fft_energies),axis=1)
		interpreters['enc0'].set_tensor(input_details['enc0'][0]['index'], input_data_enc0_stacked)
		interpreters['enc0'].invoke()
		output_data_enc0 = interpreters['enc0'].get_tensor(output_details['enc0'][0]['index'])
		
		output_data_enc0_stacked = np.concatenate((output_data_enc0_prev,output_data_enc0),axis=1)
		interpreters['enc1'].set_tensor(input_details['enc1'][0]['index'], output_data_enc0_stacked)
		interpreters['enc1'].invoke()
		output_data_enc1 = interpreters['enc1'].get_tensor(output_details['enc1'][0]['index'])
		
		interpreters['joint'].set_tensor(input_details['joint'][0]['index'], output_data_dec)
		interpreters['joint'].set_tensor(input_details['joint'][1]['index'], output_data_enc1)
		interpreters['joint'].invoke()
		output_data_joint = interpreters['joint'].get_tensor(output_details['joint'][0]['index'])
		
		interpreters['dec'].set_tensor(input_details['dec'][0]['index'], output_data_joint)
		interpreters['dec'].invoke()
		output_data_dec = interpreters['dec'].get_tensor(output_details['dec'][0]['index'])


		# roll the stackers for next iteration
		fft_energies_prevprev = fft_energies_prev
		fft_energies_prev = fft_energies
		output_data_enc0_prev = output_data_enc0

		# prevent NaNs in the dec output, the loop will not recover from that
		output_data_dec[np.isnan(output_data_dec)] = 0


		# feed the output from the decoder to the symbol FST
		# print(input_data_enc0_stacked)
		print(output_data_dec)


print(np.histogram(list_a))
print(np.histogram(list_b))
print(np.histogram(list_c))

f, (wav,plota,plotb,plotc) = plt.subplots(4, sharex=True)
wav.plot(np.linspace(0, len(input_buffer_full)/float(samplerate), len(input_buffer_full), endpoint=False), input_buffer_full)
plota.plot(np.linspace(0, len(input_buffer_full)/float(samplerate), ((len(input_buffer_full)/float(samplerate))*100) -2), list_a)
plotb.plot(np.linspace(0, len(input_buffer_full)/float(samplerate), ((len(input_buffer_full)/float(samplerate))*100) -2), list_b)
plotc.plot(np.linspace(0, len(input_buffer_full)/float(samplerate), ((len(input_buffer_full)/float(samplerate))*100) -2), list_c)
plt.show()