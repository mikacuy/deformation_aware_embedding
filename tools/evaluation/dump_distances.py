from ctypes import *
import os
import sys

libpath = os.path.abspath(__file__)
l = len(libpath.split('/')[-1])
libpath = libpath[:-l] + 'build/'

distances = cdll.LoadLibrary(libpath + '/libdistances.so')
distances.OneWayChamfer.restype = c_double
distances.TwoWayChamfer.restype = c_double

def cpath(str):
	return c_char_p(str.encode('utf-8'))

src_file = sys.argv[1]
tar_file = sys.argv[2]
distance_type = int(sys.argv[3])
out_file = sys.argv[4]
distances.LoadObj(cpath(src_file), 1) # cad to scan distance is actually larger
distances.LoadObj(cpath(tar_file), 2) # scan to cad distance is smaller

if distance_type == 1:
	dis = distances.OneWayChamfer(1,50000)
else:
	dis = distances.TwoWayChamfer(50000)

fp = open(out_file, 'w')
fp.write('%lf\n'%(dis))
fp.close()