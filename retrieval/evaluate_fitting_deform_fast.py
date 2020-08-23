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

###For parallel running
import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

np.random.seed(0)
start_time = time.time()

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
parser.add_argument('--category', default='chair', help='Which class')
parser.add_argument('--data_split', default = "test", help='which data split to use')
parser.add_argument('--dump_dir', default='dump_chair_autoencoder/', help='dump folder path [dump]')
parser.add_argument('--fitting_dump_dir', default='deformation_parallel_newcost/', help='dump folder path after fitting')
parser.add_argument('--to_deform', default=True, help='with or without deformation')
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

if (OBJ_CAT == "car" or OBJ_CAT == "airplane"):
    with open('../shapenetcore_v2_split2.json') as json_file:
        data = json.load(json_file)    

TO_DEFORM = FLAGS.to_deform
print("Deform "+str(TO_DEFORM))

if not TO_DEFORM:
	print("ERROR. Please run evaluate_fitting.py instead.")
	exit()

if (TO_DEFORM):
	output_model_fol = os.path.join(FITTING_DUMP_DIR, "models")
	output_image_fol = os.path.join(FITTING_DUMP_DIR, "images")
	if not os.path.exists(output_model_fol):
		os.mkdir(output_model_fol)		
	if not os.path.exists(output_image_fol):
		os.mkdir(output_image_fol)

	output_txt_fol = os.path.join(FITTING_DUMP_DIR, "txt_cost")
	if not os.path.exists(output_txt_fol):
		os.mkdir(output_txt_fol)	

##For coverage error
coverage_txt_fol = os.path.join(FITTING_DUMP_DIR, "coverage_cost")
if not os.path.exists(coverage_txt_fol):
	os.mkdir(coverage_txt_fol)

### Already selected models that were previously deformed
SOURCE_FOLDER_LIST = []

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

#####Get indices of neighbors in the sources
already_computed_neighbors = []
for k in range(len(SOURCE_FOLDER_LIST)):
    pickle_in = open(os.path.join(SOURCE_FOLDER_LIST[k], "neighbors.pickle"), "rb")
    nb_idx = pickle.load(pickle_in)
    already_computed_neighbors.append(nb_idx)


shapes = data[str(cat_idx)]
synsetid = shapes["synsetid"]
model_names = shapes["model_names"]
num_samples = shapes["num_samples"]

##########Generate sh file for parallel deformation
script_fol = os.path.join(FITTING_DUMP_DIR, "scripts_deform/")
if not os.path.exists(script_fol):
	os.mkdir(script_fol)

filename = OBJ_CAT+"_"+DATA_SPLIT+".sh"
f = open(os.path.join(script_fol, filename), "w")

# for i in range(len(model_names)):
for i in range(2):
    ref_model_name = model_names[i]
    ref_filename = os.path.join(SHAPENET_BASEDIR, synsetid, ref_model_name, 'models', 'model_normalized.obj')

    for j in range(NUM_NEIGHBORS):
        # src_model_name = model_names[neighbors_idxs[i][j]]
        # src_filename = os.path.join(SHAPENET_BASEDIR, synsetid, src_model_name, 'models', 'model_normalized.obj')

        output_filename_raw = ref_model_name + "_" + str(j) + ".obj"
        output_filename = os.path.join(output_model_fol, output_filename_raw)
        textfile_name = os.path.join(output_txt_fol, ref_model_name + "_" + str(j) +'_cost.txt')
        output_image_filename = os.path.join(output_image_fol, ref_model_name + "_" + str(j) + ".png")
        coverage_textfile_name = os.path.join(coverage_txt_fol, ref_model_name + "_" + str(j) +'_coverage_cost.txt')

        found = False
        ###Check if already computed before
        for k in range(len(SOURCE_FOLDER_LIST)):
            curr_nb = already_computed_neighbors[k][i]

            if not ((neighbors_idxs[i][j]) in curr_nb):
                continue

            for nb_idx in range(len(curr_nb)):
                if (neighbors_idxs[i][j]) == curr_nb[nb_idx]:
                    src_folder_dump_dir = os.path.join(SOURCE_FOLDER_LIST[k], FLAGS.fitting_dump_dir)
                    src_obj_filename = os.path.join(src_folder_dump_dir, "models", ref_model_name + "_" + str(nb_idx) + ".obj")

                    ##copy files
                    src_textfile_name = os.path.join(src_folder_dump_dir, "txt_cost", ref_model_name + "_" + str(nb_idx) +'_cost.txt')
                    src_output_image_filename = os.path.join(src_folder_dump_dir, "images", ref_model_name + "_" + str(nb_idx) + ".png")
                    src_coverage_textfile_name = os.path.join(src_folder_dump_dir, "coverage_cost", ref_model_name + "_" + str(nb_idx) +'_coverage_cost.txt')                

                    os.system('cp %s %s' % (src_obj_filename, output_filename))
                    os.system('cp %s %s' % (src_textfile_name, textfile_name))
                    os.system('cp %s %s' % (src_output_image_filename , output_image_filename))
                    os.system('cp %s %s' % (src_coverage_textfile_name, coverage_textfile_name))

                    found = True
                    break
                else:
                    continue

            if (found):
                break

        if not found:
            src_model_name = model_names[neighbors_idxs[i][j]]
            src_filename = os.path.join(SHAPENET_BASEDIR, synsetid, src_model_name, 'models', 'model_normalized.obj')    
            command = "python deform_retrieval_single_pair.py " + output_filename + " " + ref_filename + " " + src_filename + " " + textfile_name + " " + output_image_filename + " " + coverage_textfile_name
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

os.system("python evaluate.py --dump_dir=" + (str(DUMP_DIR).encode('ascii', 'ignore')) + " --category="+str(OBJ_CAT))
os.system("python evaluate_point2mesh.py --dump_dir=" + (str(DUMP_DIR).encode('ascii', 'ignore')) + " --category="+ str(OBJ_CAT)+ " --fitting_dump_dir=point2mesh_new2_deform/ --to_deform=1")
print("Time elapsed: "+str(time.time()-start_time)+" sec.")



