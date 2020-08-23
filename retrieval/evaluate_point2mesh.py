import os
import sys
import glob
import numpy as np
import random
import json
import argparse
import pickle
import time
import h5py
import math

import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

np.random.seed(0)
start_time = time.time()

def get_all_loss(folder):
	files = glob.glob(folder + '/*.txt')
	errs = []
	for f in files:
		lines = [l.strip() for l in open(f)]
		errs.append(float(lines[0]))
	return np.array(errs)

with open('../shapenetcore_v2_split.json') as json_file:
    data = json.load(json_file)

SHAPENET_BASEDIR = '/orion/group/ShapeNetManifold_10000_simplified/'

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='chair', help='Which class')
parser.add_argument('--data_split', default = "test", help='which data split to use')

parser.add_argument('--dump_dir', default='dump_chair_ranked_cd/', help='dump folder path [dump]')

parser.add_argument('--fitting_dump_dir', default='point2mesh_new2_nodeform/', help='dump folder path after fitting')
parser.add_argument('--to_deform', default=False, help='with or without deformation')
parser.add_argument('--num_neighbors', type=int, default=3, help='Number of neighbors to retrieve')

FLAGS = parser.parse_args() 
OBJ_CAT = FLAGS.category
DATA_SPLIT = FLAGS.data_split
NUM_NEIGHBORS = FLAGS.num_neighbors
DUMP_DIR = str(FLAGS.dump_dir)
print(DUMP_DIR)

FITTING_DUMP_DIR = os.path.join(DUMP_DIR, FLAGS.fitting_dump_dir)
if not os.path.exists(FITTING_DUMP_DIR): os.mkdir(FITTING_DUMP_DIR)
LOG_FOUT = open(os.path.join(FITTING_DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n') 

TO_DEFORM = FLAGS.to_deform
print("Deform "+str(TO_DEFORM))

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

shapes = data[str(cat_idx)]
synsetid = shapes["synsetid"]
model_names = shapes["model_names"]
num_samples = shapes["num_samples"]

#Retrieved neighbor indices
pickle_in = open(os.path.join(DUMP_DIR, "neighbors.pickle"),"rb")
neighbors_idxs = pickle.load(pickle_in)

##########Generate sh file for parallel deformation
script_fol = os.path.join(FITTING_DUMP_DIR, "scripts_deform/")
if not os.path.exists(script_fol):
	os.mkdir(script_fol)

script_filename = OBJ_CAT+"_"+DATA_SPLIT+".sh"
f = open(os.path.join(script_fol, script_filename), "w")

# Gen Scripts
temp_fol = os.path.join(FITTING_DUMP_DIR, 'temp')
if not os.path.exists(temp_fol):
	os.mkdir(temp_fol)
for i in range(NUM_NEIGHBORS):
	rank_fol = os.path.join(temp_fol, "rank_"+str(i))
	if not os.path.exists(rank_fol):
		os.mkdir(rank_fol)	

found_idx = []
# for i in range(len(model_names)):
for i in range(2):
	ref_model_name = model_names[i]
	ref_filename = os.path.join(SHAPENET_BASEDIR, synsetid, ref_model_name, 'models', 'model_normalized.obj')

	# if (i==146):
	# 	continue

	commands = []
	got_error = False
	for j in range(NUM_NEIGHBORS):
		# cost_txt_filename = os.path.join(temp_fol, "rank_"+str(j), ref_model_name+".txt")
		cost_txt_filename = os.path.join(temp_fol, "rank_"+str(j), str(i)+".txt")
		
		fol_name = os.path.join(DUMP_DIR, 'deformation_parallel_newcost', 'models')
		filename = os.path.join(fol_name, ref_model_name + "_" + str(j) + ".obj")

		if (TO_DEFORM):
			if not os.path.exists(filename):
				print("Filename does not exist.")
				got_error = True
				commands=[]
				break			
			command = "python ../tools/evaluation/dump_distances.py " + filename + " " + ref_filename + " 2 " + cost_txt_filename + "\n"				

		else:
			src_model_name = model_names[neighbors_idxs[i][j]]
			src_filename = os.path.join(SHAPENET_BASEDIR, synsetid, src_model_name, 'models', 'model_normalized.obj')

			if not os.path.exists(src_filename):
				print("ShapeNet manifold does not exist")
				got_error = True
				commands =[]
				break

			command = "python ../tools/evaluation/dump_distances.py " + src_filename + " " + ref_filename + " 2 " + cost_txt_filename + "\n"

		commands.append(command)

		if (j==NUM_NEIGHBORS-1):
			found_idx.append(i)

	if (got_error):
		continue

	##Write to file
	for command in commands:
		f.write(command)

	if (i%50==0):
		print("Time elapsed: "+str(time.time()-start_time)+" sec for "+str(i)+"/"+str(num_samples)+" samples.")
f.close()

######Run deformations in parallel#####
command_file = os.path.join(script_fol, script_filename)
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

cd_errors = []
for i in range(NUM_NEIGHBORS):
	print(get_all_loss(os.path.join(temp_fol, "rank_"+str(i))).shape)
	cd_errors.append(get_all_loss(os.path.join(temp_fol, "rank_"+str(i))))

cd_errors = np.array(cd_errors)
cd_errors = cd_errors.T
print(cd_errors.shape)

for i in range(NUM_NEIGHBORS):
	i_mean_cd = np.mean(cd_errors[:,i])
	log_string("Rank "+ str(i+1) + " retrieved mean CD error: "+str(i_mean_cd))

mean_cd = np.mean(np.mean(cd_errors, axis=1))
log_string("Average CD error: "+str(mean_cd))
log_string(" ")

print(np.min(cd_errors, axis=1).shape)
min_cd = np.mean(np.min(cd_errors, axis=1)) 
log_string("Average minimum CD of top "+str(NUM_NEIGHBORS)+": "+str(min_cd))

log_string(" ")

print("Total running time: "+str(time.time()-start_time)+" sec")
LOG_FOUT.close()

