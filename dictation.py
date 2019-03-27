#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import marisa_trie
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
stfts = tf.signal.stft(signals, frame_length=samples_per_25ms, frame_step=samples_per_10ms) # does hannig window by default
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

# init the symbol table
trie = marisa_trie.Trie()
trie.load('syms.marisa')

# init models
models = ['ep','enc0','enc1','dec','joint']
interpreters = {}
input_details = {}
tensor_details = {}
output_details = {}

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



# init the loop in the decoder, the paper indicates a start-of-sentence <sos> should be provided
sym_prob_shape = output_details['joint'][0]['shape']
sym_prob = np.zeros(sym_prob_shape, dtype=np.float32)
sym_prob[0][trie[u'<sorw>']] = 1

# run over frames
filterbank_energies_stack = []
log_mel_spectrograms_40_80 = zip(log_mel_spectrograms_40[0],log_mel_spectrograms_80[0])
for filterbank_energies_ep,filterbank_energies in log_mel_spectrograms_40_80:
	filterbank_energies_stack.append(filterbank_energies)

	# run the endpointer to decide if we should run the RNN
	input_data_ep = [filterbank_energies_ep]
	interpreters['ep'].set_tensor(input_details['ep'][0]['index'], input_data_ep)
	interpreters['ep'].invoke()
	output_data_ep = interpreters['ep'].get_tensor(output_details['ep'][0]['index'])
	[[P_speech,P_nonspeech]] = output_data_ep


	# feed the RNN
	if len(filterbank_energies_stack) == 6:
		input_data_enc0_stack1 = np.concatenate(([filterbank_energies_stack[0]],[filterbank_energies_stack[1]],[filterbank_energies_stack[2]]),axis=1)
		interpreters['enc0'].set_tensor(input_details['enc0'][0]['index'], input_data_enc0_stack1)
		interpreters['enc0'].invoke()
		output_data_enc0_1 = interpreters['enc0'].get_tensor(output_details['enc0'][0]['index'])

		input_data_enc0_stack2 = np.concatenate(([filterbank_energies_stack[3]],[filterbank_energies_stack[4]],[filterbank_energies_stack[5]]),axis=1)
		interpreters['enc0'].set_tensor(input_details['enc0'][0]['index'], input_data_enc0_stack2)
		interpreters['enc0'].invoke()
		output_data_enc0_2 = interpreters['enc0'].get_tensor(output_details['enc0'][0]['index'])
		
		output_data_enc0_stacked = np.concatenate((output_data_enc0_1,output_data_enc0_2),axis=1)
		interpreters['enc1'].set_tensor(input_details['enc1'][0]['index'], output_data_enc0_stacked)
		interpreters['enc1'].invoke()
		output_data_enc1 = interpreters['enc1'].get_tensor(output_details['enc1'][0]['index'])
		
		# the decoder is fed with the symbol probabilities of the previous iteration
		interpreters['dec'].set_tensor(input_details['dec'][0]['index'], sym_prob)
		interpreters['dec'].invoke()
		output_data_dec = interpreters['dec'].get_tensor(output_details['dec'][0]['index'])
		
		interpreters['joint'].set_tensor(input_details['joint'][0]['index'], output_data_dec)
		interpreters['joint'].set_tensor(input_details['joint'][1]['index'], output_data_enc1)
		interpreters['joint'].invoke()
		output_data_joint = interpreters['joint'].get_tensor(output_details['joint'][0]['index'])

		# softmax the output of the joint
		sym_prob = tf.nn.softmax(output_data_joint)

		# feed the output from the softmax to the symbol table
		max_prob_char = tf.argmax(sym_prob,axis=1)[0].numpy()
		print(trie.restore_key(max_prob_char))

		del filterbank_energies_stack[:]