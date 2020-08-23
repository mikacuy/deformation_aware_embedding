import argparse
import math
from datetime import datetime
import h5py
import numpy as np

import socket
import importlib
import os
import sys
import json
import pickle

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath + '/../..')
import objloader
import mesh_utils

import tensorflow as tf
sys.path.append(os.path.join(libpath, '../tf_ops/nn_distance'))
import tf_nndistance

import time
from pdb import set_trace

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='chair', help='Which class')
parser.add_argument('--data_split', default = "train", help='which data split to use')
FLAGS = parser.parse_args()
OBJ_CAT = FLAGS.category
DATA_SPLIT = FLAGS.data_split

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

#### Get point clouds for each object instance
with open('../../shapenetcore_v2_split.json') as json_file:
    data = json.load(json_file)

train_data = data[DATA_SPLIT]
num_categories = len(list(train_data.keys()))

idx = -1
cat_name = ""
for i in range(num_categories):
    cat_name = train_data[str(i)]["category"]
    if (cat_name == OBJ_CAT):
        idx = i
        break

synsetid = train_data[str(idx)]["synsetid"]
model_names = train_data[str(idx)]["model_names"]
num_samples = train_data[str(idx)]["num_samples"]

print("Category Name: "+cat_name)
print("Synsetid: "+synsetid)
print("Num samples: "+str(num_samples))


NUM_POINT = 2048
NUM_CANDIDATES = 50

####Get candidates
pos_pickle_in = open('../../candidate_generation/candidates_'+DATA_SPLIT+'_'+OBJ_CAT+'_retrieval.pickle',"rb")
pos_candidate_idxs = pickle.load(pos_pickle_in)
POSITIVE_CANDIDATES_FOL = '/orion/downloads/deformation_aware_embedding/manifold_top50_cd_candidates/'
pos_pickle_in.close()

neg_pickle_in = open('../../candidate_generation/candidates_'+DATA_SPLIT+'_'+OBJ_CAT+'_negatives.pickle',"rb")
neg_candidate_idxs = pickle.load(neg_pickle_in)
neg_pickle_in.close()
NEGATIVE_CANDIDATES_FOL = '/orion/downloads/deformation_aware_embedding/manifold_negatives_cd_candidates/'

def chamfer_loss(pc1, pc2):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pc1, pc2)
    loss = tf.reduce_mean(dists_forward+dists_backward, axis=1)
    return loss

with tf.Graph().as_default():
	with tf.device('/gpu:1'):
	    pointclouds_pl_1 = tf.placeholder(tf.float32, shape=(NUM_CANDIDATES, NUM_POINT, 3))
	    pointclouds_pl_2 = tf.placeholder(tf.float32, shape=(NUM_CANDIDATES, NUM_POINT, 3))

	    chamfer_distance = chamfer_loss(pointclouds_pl_1, pointclouds_pl_2)
	    
	# Create a session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)

	# Init variables
	init = tf.global_variables_initializer()
	sess.run(init)

	ops = {'pointclouds_pl_1': pointclouds_pl_1,
	       'pointclouds_pl_2': pointclouds_pl_2,
	       'chamfer_distance': chamfer_distance,
	        }

	fname = '../../candidate_generation/' + DATA_SPLIT +"_"+OBJ_CAT + '.h5'
	OBJ_POINTCLOUDS = load_h5(fname)
	print(OBJ_POINTCLOUDS.shape)	        

	positive_chamfer_costs = []
	negative_chamfer_costs = []

	# for i in range(20):
	for i in range(len(model_names)):
		ref_model_name = model_names[i]
		pc_ref = OBJ_POINTCLOUDS[i]
		all_pc_ref = []
		for _ in range(NUM_CANDIDATES):
			all_pc_ref.append(pc_ref)
		all_pc_ref = np.array(all_pc_ref)		


		##########Positives
		output_model_fol = os.path.join(POSITIVE_CANDIDATES_FOL, cat_name, ref_model_name)
		all_pc_src = []

		non_existing_pos_idx = []
		for j in range(len(pos_candidate_idxs[0])):
			src_model_name = model_names[pos_candidate_idxs[i][j]]
			src_filename = os.path.join(output_model_fol, src_model_name + "_" + ref_model_name+'.obj')
			try:
				V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(src_filename)
				pc_src = mesh_utils.mesh_to_pc(V, F, num_points=NUM_POINT)
			except:
				print(src_filename+" does not exist from model "+str(i))
				non_existing_pos_idx.append(j)
				pc_src = np.zeros((NUM_POINT, 3))

			all_pc_src.append(pc_src)
		all_pc_src = np.array(all_pc_src)

		feed_dict = {ops['pointclouds_pl_1']: all_pc_ref,
		         ops['pointclouds_pl_2']: all_pc_src,}
		chamfer_distance = sess.run([ops['chamfer_distance']], feed_dict=feed_dict)
		chamfer_distance = np.array(chamfer_distance)[0]

		if (len(non_existing_pos_idx)>0):
			non_existing_pos_idx = np.array(non_existing_pos_idx)
		chamfer_distance[non_existing_pos_idx] = -1
		
		positive_chamfer_costs.append(chamfer_distance)

		#######Negatives
		output_model_fol = os.path.join(NEGATIVE_CANDIDATES_FOL, cat_name, ref_model_name)
		all_pc_src = []

		non_existing_neg_idx = []
		for j in range(len(neg_candidate_idxs[0])):
			src_model_name = model_names[neg_candidate_idxs[i][j]]
			src_filename = os.path.join(output_model_fol, src_model_name + "_" + ref_model_name+'.obj')
			try:
				V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(src_filename)
				pc_src = mesh_utils.mesh_to_pc(V, F, num_points=NUM_POINT)
			except:
				print(src_filename+" does not exist from model "+str(i))
				non_existing_neg_idx.append(j)
				pc_src = np.zeros((NUM_POINT, 3))

			all_pc_src.append(pc_src)
		all_pc_src = np.array(all_pc_src)

		feed_dict = {ops['pointclouds_pl_1']: all_pc_ref,
		         ops['pointclouds_pl_2']: all_pc_src,}
		chamfer_distance = sess.run([ops['chamfer_distance']], feed_dict=feed_dict)
		chamfer_distance = np.array(chamfer_distance)[0]

		if (len(non_existing_neg_idx)>0):
			non_existing_neg_idx = np.array(non_existing_neg_idx)
		chamfer_distance[non_existing_neg_idx] = -1	

		negative_chamfer_costs.append(chamfer_distance)	

		if (i%100==0):
			print("Time elapsed: "+str(time.time()-start_time)+" sec for "+str(i)+" samples.")


	positive_chamfer_costs = np.array(positive_chamfer_costs)
	negative_chamfer_costs = np.array(negative_chamfer_costs)

	print(positive_chamfer_costs.shape)
	print(negative_chamfer_costs.shape)

	OUTPUT_DIR = "../chamfer_distance_deformed_candidates/"
	if not os.path.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

	filename = os.path.join(OUTPUT_DIR, "positive_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+".pickle")
	with open(filename, 'w') as handle:
		pickle.dump(positive_chamfer_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)

	filename = os.path.join(OUTPUT_DIR, "negative_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+".pickle")
	with open(filename, 'w') as handle:
		pickle.dump(negative_chamfer_costs, handle, protocol=pickle.HIGHEST_PROTOCOL)








