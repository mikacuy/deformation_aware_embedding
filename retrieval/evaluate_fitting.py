import os
import sys
import numpy as np
import random
import json
import argparse
import pickle
libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath + '/../pyRender/lib')
sys.path.append(libpath + '/../pyRender/src')
sys.path.append(libpath + '/..')
import render
import objloader
import mesh_utils
import time
import h5py
import math

import skimage.io as sio
from PIL import Image

###For parallel running
import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

np.random.seed(0)
start_time = time.time()
THRESHOLD = 1e-3;

#For renderer
info = {'Height':480, 'Width':640, 'fx':2000, 'fy':2000, 'cx':319.5, 'cy':239.5}
render.setup(info)
cam2world = np.array([[ 0.85408425,  0.31617427, -0.375678  ,  0.56351697 * 2],
	   [ 0.        , -0.72227067, -0.60786998,  0.91180497 * 2],
	   [-0.52013469,  0.51917219, -0.61688   ,  0.92532003 * 2],
	   [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)

world2cam = np.linalg.inv(cam2world).astype('float32')
total_width = 3*info["Width"]
height = info["Height"]	
new_im = Image.new('L', (total_width, height))

def chamfer_distance(root, sample):
    """
    arguments: 
        root: the array, size: (num_point, num_feature)
        sample: the samples, size: (num_point, num_feature)
        each entry is the distance from a sample to root
    returns:
        distances: one way chamfer distance of the source (src) to the reference (ref)

    To get the coverage of the reference,
    : Need to calculate chamfer_distance(deformed_model, reference) 
    : how well reference is represented by the model
    """
    num_point, num_features = root.shape
    expanded_root = np.tile(root, (num_point, 1))
    expanded_sample = np.reshape(
            np.tile(np.expand_dims(sample, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_root-expanded_sample, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.sum(distances)
    return distances

##############h5 file handles
def save_dataset(fname, pcs):
    cloud = np.stack([pc for pc in pcs])

    fout = h5py.File(fname)
    fout.create_dataset('data', data=cloud, compression='gzip', dtype='float32')
    fout.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    return data
################################

with open('../shapenetcore_v2_split.json') as json_file:
    data = json.load(json_file)

SHAPENET_BASEDIR = '/orion/group/ShapeNetManifold_10000_simplified/'

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='table', help='Which class')
parser.add_argument('--data_split', default = "test", help='which data split to use')
parser.add_argument('--dump_dir', default='dump_retrieval_tables_autoencoder/', help='dump folder path [dump]')
parser.add_argument('--fitting_dump_dir', default='no_deformation_parallel_newcost/', help='dump folder path after fitting')
parser.add_argument('--to_deform', default=False, help='with or without deformation')
parser.add_argument('--num_neighbors', type=int, default=3, help='Number of neighbors to retrieve')

FLAGS = parser.parse_args()
OBJ_CAT = FLAGS.category
DATA_SPLIT = FLAGS.data_split
NUM_NEIGHBORS = FLAGS.num_neighbors
DUMP_DIR = FLAGS.dump_dir

FITTING_DUMP_DIR = os.path.join(DUMP_DIR, FLAGS.fitting_dump_dir)
if not os.path.exists(FITTING_DUMP_DIR): os.mkdir(FITTING_DUMP_DIR)
LOG_FOUT = open(os.path.join(FITTING_DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

TO_DEFORM = FLAGS.to_deform
print("Deform "+str(TO_DEFORM))

if TO_DEFORM:
	print("ERROR. Please run evaluate_fitting_deform.py instead.")
	exit()

##For coverage error
coverage_txt_fol = os.path.join(FITTING_DUMP_DIR, "coverage_cost")
if not os.path.exists(coverage_txt_fol):
	os.mkdir(coverage_txt_fol)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

data = data[DATA_SPLIT]
num_categories = len(list(data.keys()))
cat_idx = -1
for i in range(num_categories):
	if (data[str(i)]["category"] == OBJ_CAT):
		cat_idx = str(i) 
		break

#Retrieved neighbor indices
pickle_in = open(os.path.join(DUMP_DIR, "neighbors.pickle"),"rb")
neighbors_idxs = pickle.load(pickle_in)

shapes = data[str(cat_idx)]
synsetid = shapes["synsetid"]
model_names = shapes["model_names"]
num_samples = shapes["num_samples"]

##########Generate sh file for parallel cost calculation
script_fol = os.path.join(FITTING_DUMP_DIR, "scripts_no_deform/")
if not os.path.exists(script_fol):
	os.mkdir(script_fol)

filename = OBJ_CAT+"_"+DATA_SPLIT+".sh"
f = open(os.path.join(script_fol, filename), "w")

for i in range(len(model_names)):
	ref_model_name = model_names[i]
	ref_filename = os.path.join(SHAPENET_BASEDIR, synsetid, ref_model_name, 'models', 'model_normalized.obj')	

	for j in range(NUM_NEIGHBORS):
		src_model_name = model_names[neighbors_idxs[i][j]]
		src_filename = os.path.join(SHAPENET_BASEDIR, synsetid, src_model_name, 'models', 'model_normalized.obj')

		coverage_textfile_name = os.path.join(coverage_txt_fol, ref_model_name + "_" + str(j) +'_coverage_cost.txt')
		command = "../meshdeform/build_cost_new/deform " + src_filename + " " + ref_filename + " " + "dummy" + " " + coverage_textfile_name
		f.write(command+'\n')
f.close()
##############

######Run deformations in parallel#####
command_file = os.path.join(script_fol, filename)
commands = [line.rstrip() for line in open(command_file)]
print("Number of commands in parallel: "+str(len(commands)))

report_step = 8
pool = Pool(report_step)
for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
  if idx % report_step == 0:
     print('[%s] command %d of %d' % (datetime.datetime.now().time(), idx, len(commands)))
  if return_code != 0:
     print('!! command %d of %d (\"%s\") failed' % (idx, len(commands), commands[idx]))
################


# ##To sample points from mesh
# OBJ_POINTCLOUDS = []
fname = DATA_SPLIT +"_"+OBJ_CAT+"_meshsampled.h5"
pcs = load_h5(fname)
OBJ_POINTCLOUDS = load_h5(fname)
print(OBJ_POINTCLOUDS.shape)

cd_errors = []
coverage_errors = []

for i in range(len(model_names)):
	ref_model_name = model_names[i]
	ref_filename = os.path.join(SHAPENET_BASEDIR, synsetid, ref_model_name, 'models', 'model_normalized.obj')
	# V_ref, F_ref, _, _, _, _, _, _ = objloader.LoadSimpleOBJ_manifold(ref_filename)
	# pc_ref = mesh_utils.mesh_to_pc(V_ref, F_ref, num_points=2000)

	pc_ref = OBJ_POINTCLOUDS[i]
	# OBJ_POINTCLOUDS.append(pc_ref)		

	cd_error = []
	coverage_error = []
	for j in range(NUM_NEIGHBORS):
		src_model_name = model_names[neighbors_idxs[i][j]]
		src_filename = os.path.join(SHAPENET_BASEDIR, synsetid, src_model_name, 'models', 'model_normalized.obj')

		### Without Deformation
		if not TO_DEFORM:

			##For CD error
			# V_src, F_src, _, _, _, _, _, _ = objloader.LoadSimpleOBJ_manifold(src_filename)
			# pc_src = mesh_utils.mesh_to_pc(V_src, F_src, num_points=2000)
			pc_src = OBJ_POINTCLOUDS[neighbors_idxs[i][j]]
			error_CD = chamfer_distance(pc_src, pc_ref)

			# ##For coverage error
			# textfile_name = os.path.join(coverage_txt_fol, ref_model_name + "_" + str(j) +'_coverage_cost.txt')
			# ###Get coverage cost
			# command = "../meshdeform/build_cost/deform " + src_filename + " " + ref_filename + " " + "dummy" + " " + textfile_name
			# os.system(command)

			###Read coverage threshold
			coverage_textfile_name = os.path.join(coverage_txt_fol, ref_model_name + "_" + str(j) +'_coverage_cost.txt')
			f = open(coverage_textfile_name, "r")
			error_coverage = float(f.readlines()[0].split('\t')[1])
			f.close()		

		cd_error.append(error_CD)
		coverage_error.append(error_coverage)

	cd_errors.append(cd_error)
	coverage_errors.append(coverage_error)

	if (i%50==0):
		print("Time elapsed: "+str(time.time()-start_time)+" sec for "+str(i)+"/"+str(num_samples)+" samples.")


# ####Save sampled point clouds from mesh
# OBJ_POINTCLOUDS = np.array(OBJ_POINTCLOUDS)
# fname = DATA_SPLIT +"_"+OBJ_CAT+"_meshsampled.h5"
# save_dataset(fname, OBJ_POINTCLOUDS)
# pcs = load_h5(fname)
# OBJ_POINTCLOUDS = load_h5(fname)
# print(OBJ_POINTCLOUDS.shape)
# #######

cd_errors = np.array(cd_errors)
coverage_errors = np.array(coverage_errors)
print(cd_errors.shape)

for i in range(NUM_NEIGHBORS):
	i_mean_cd = np.mean(cd_errors[:,i])
	i_mean_coverage = np.mean(coverage_errors[:,i])
	log_string("Rank "+ str(i) + " retrieved mean CD error: "+str(i_mean_cd))
	log_string("Rank "+ str(i) + " retrieved mean coverage error: "+str(i_mean_coverage))

mean_cd = np.mean(np.mean(cd_errors, axis=1))
mean_coverage = np.mean(np.mean(coverage_errors, axis=1))
log_string("Average CD error: "+str(mean_cd))
log_string("Average coverage error: "+str(mean_coverage))
print("Total running time: "+str(time.time()-start_time)+" sec")
LOG_FOUT.close()
