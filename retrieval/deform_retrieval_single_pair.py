import os
import sys
import numpy as np
import random
import json
import argparse
import pickle
libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath + '/..')
import objloader
import mesh_utils
import time
import h5py
import math

##### Need to pass arguement output_filename, ref_filename, src_filename, textfile_name, output_image_filename
output_filename = sys.argv[1]
ref_filename = sys.argv[2]
src_filename = sys.argv[3]
textfile_name = sys.argv[4]
output_image_filename = sys.argv[5]
coverage_textfile_name = sys.argv[6]

command = "../meshdeform/build/deform " + src_filename + " " + ref_filename + " " + output_filename + " " + textfile_name
os.system(command)

#Get coverage cost
command = "../meshdeform/build/getcost " + output_filename + " " + ref_filename + " " + "dummy" + " " + coverage_textfile_name
os.system(command)

### DEBUGGING FILES 
# sys.path.append(libpath + '/../pyRender/lib')
# sys.path.append(libpath + '/../pyRender/src')
# import render
# import skimage.io as sio
# from PIL import Image

# THRESHOLD = 1e-3;
###For renderer
# info = {'Height':480, 'Width':640, 'fx':2000, 'fy':2000, 'cx':319.5, 'cy':239.5}
# render.setup(info)
# cam2world = np.array([[ 0.85408425,  0.31617427, -0.375678  ,  0.56351697 * 2],
# 	   [ 0.        , -0.72227067, -0.60786998,  0.91180497 * 2],
# 	   [-0.52013469,  0.51917219, -0.61688   ,  0.92532003 * 2],
# 	   [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)

# world2cam = np.linalg.inv(cam2world).astype('float32')
# total_width = 3*info["Width"]
# height = info["Height"]	
# new_im = Image.new('L', (total_width, height))

# ###See if valid ARAP threshold
# f = open(textfile_name, "r")
# cost = float(f.readlines()[0].split('\t')[1])

# ##########Render with pyrender
# V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(ref_filename)
# context = render.SetMesh(V, F)
# render.render(context, world2cam)
# depth_ref = render.getDepth(info)
# depth_ref /= np.max(depth_ref)
# render.Clear()	

# V_src, F_src, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(output_filename)
# context = render.SetMesh(V_src, F_src)
# render.render(context, world2cam)
# depth_deformed = render.getDepth(info)
# depth_deformed /= np.max(depth_deformed)

# if (cost > THRESHOLD):
# 	depth_deformed[np.where(depth_deformed==0)] = 100.0

# render.Clear()

# V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(src_filename)
# context = render.SetMesh(V, F)
# render.render(context, world2cam)
# depth_src = render.getDepth(info)
# depth_src /= np.max(depth_src)
# render.Clear()

# ##Convert to images and combine
# img_deformed = Image.fromarray(np.uint8(depth_deformed*255), 'L')
# img_src = Image.fromarray(np.uint8(depth_src*255), 'L')
# img_ref = Image.fromarray(np.uint8(depth_ref*255), 'L')
# images = [img_src, img_ref, img_deformed]

# x_offset = 0
# for im in images:
# 	new_im.paste(im, (x_offset,0))
# 	x_offset += im.size[0]

# new_im.save(output_image_filename)



