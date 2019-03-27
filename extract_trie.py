#!/usr/bin/env python
import sys
fname = sys.argv[1]
b = bytearray(open(fname, 'rb').read())
for i in range(len(b)):
	if b[i:i+15] == "We love Marisa.":
		open(fname + '.marisa', 'wb').write(b[i:])
		break