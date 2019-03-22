#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy.io.wavfile

samplerate,rawdata = scipy.io.wavfile.read('sample.wav')
input_buffer_full = rawdata[:,0]	# probably two channels, 0=left,1=right
samples_per_10ms = samplerate /100
samples_per_25ms = samplerate /40

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
fft_energies_prev = np.empty([1, 80], dtype=np.float32)
fft_energies_prevprev = fft_energies_prev
output_shape_enc0 = output_details['enc0'][0]['shape']
output_data_enc0_prev = np.array(np.random.random_sample(output_shape_enc0), dtype=np.float32)

# init the loop in the decoder
output_shape_dec = output_details['dec'][0]['shape']
output_data_dec = np.array(np.random.random_sample(output_shape_dec), dtype=np.float32)

# run over frames
for sample_iter in range(0, len(input_buffer_full) -samples_per_25ms, samples_per_10ms):
	frame = input_buffer_full[sample_iter:sample_iter +samples_per_25ms]
	frame = np.multiply(frame, np.hanning(samples_per_25ms))
	F = np.fft.fft(frame)[0:len(frame)/2]
	freq = np.fft.fftfreq(len(frame), 1/float(samplerate))[0:len(frame)/2]
	fband = [125,7500]
	channels = 80
	binwidth = (fband[1] - fband[0]) /channels
	fft_energies_at = range(fband[0] +(binwidth/2), fband[1], binwidth)
	fft_energies = np.array([np.interp(fft_energies_at, freq, np.abs(F))], dtype=np.float32)
	fft_energies /= np.max(np.abs(fft_energies), axis=1)

	# print(freq)
	# print(np.abs(F))
	# print(fft_energies_at)
	print(fft_energies)


	# run the endpointer to decide if we should run the RNN
	input_data_ep = np.array([np.mean(fft_energies.reshape(-1,2),axis=1)]) # take pair wise averages
	interpreters['ep'].set_tensor(input_details['ep'][0]['index'], input_data_ep)
	interpreters['ep'].invoke()
	output_data_ep = interpreters['ep'].get_tensor(output_details['ep'][0]['index'])
	print(input_data_ep)
	print(output_data_ep)


	# feed the RNN
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


	# feed the output from the decoder to the symbol FST
	print(output_data_dec)