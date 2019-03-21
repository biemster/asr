#!/usr/bin/env python
import tensorflow as tf

models = ['joint','dec','enc0','enc1','ep']
interpreters = {}

for m in models:
    # Load TFLite model and allocate tensors.
    interpreters[m] = tf.lite.Interpreter(model_path=m+'.tflite')
    interpreters[m].allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreters[m].get_input_details()
    output_details = interpreters[m].get_output_details()

    print(m)
    print(input_details)
    print(output_details)