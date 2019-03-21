#!/usr/bin/env python3
import sys
fname = sys.argv[1]
b = bytearray(open(fname, 'rb').read())
for i in range(len(b)): b[i] ^= 0x1a
open(fname + '.tflite', 'wb').write(b)