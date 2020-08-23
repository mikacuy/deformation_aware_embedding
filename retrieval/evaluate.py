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

import tensorflow as tf
sys.path.append(os.path.join(libpath, 'tf_ops/nn_distance'))
import tf_nndistance

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
parser.add_argument('--category', default='table', help='Which class')
parser.add_argument('--data_split', default = "test", help='which data split to use')

parser.add_argument('--dump_dir', default='dump_retrieval_table_arap_mahalanobis256_distances_objsigma2/', help='dump folder path [dump]')
parser.add_argument('--fitting_dump_dir', default='best_of_N/', help='dump folder path after fitting')
parser.add_argument('--to_deform', default=True, help='with or without deformation')
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

#Retrieved neighbor indices
pickle_in = open(os.path.join(DUMP_DIR, "neighbors.pickle"),"rb")
neighbors_idxs = pickle.load(pickle_in)

shapes = data[str(cat_idx)]
synsetid = shapes["synsetid"]
model_names = shapes["model_names"]
num_samples = shapes["num_samples"]
NUM_POINT = 2048

def chamfer_loss(pc1, pc2):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pc1, pc2)
    # loss = dists_forward+dists_backward
    loss = tf.reduce_mean(dists_forward+dists_backward, axis=1)
    return loss

with tf.Graph().as_default():
	with tf.device('/gpu:0'):
	    pointclouds_pl_1 = tf.placeholder(tf.float32, shape=(NUM_NEIGHBORS, NUM_POINT, 3))
	    pointclouds_pl_2 = tf.placeholder(tf.float32, shape=(NUM_NEIGHBORS, NUM_POINT, 3))

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
	       'chamfer_distance': chamfer_distance
	        }

	# fname = DATA_SPLIT +"_"+OBJ_CAT+"_meshsampled.h5"
	fname = '../candidate_generation/' + DATA_SPLIT +"_"+OBJ_CAT + '.h5'
	OBJ_POINTCLOUDS = load_h5(fname)
	print(OBJ_POINTCLOUDS.shape)

	cd_errors = []
	deformed_pcs = []
	# for i in range(len(model_names)):
	for i in range(2):
		ref_model_name = model_names[i]
		pc_ref = OBJ_POINTCLOUDS[i]
		all_pc_ref = []
		for _ in range(NUM_NEIGHBORS):
			all_pc_ref.append(pc_ref)
		all_pc_ref = np.array(all_pc_ref)

		all_pc_src = []
		deformed_candidates = []
		got_error = False
		for j in range(NUM_NEIGHBORS):
			if (TO_DEFORM):
				fol_name = os.path.join(DUMP_DIR, 'deformation_parallel_newcost', 'models')
				filename = os.path.join(fol_name, ref_model_name + "_" + str(j) + ".obj")

				try:
					V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(filename)
					pc_src = mesh_utils.mesh_to_pc(V, F, num_points=NUM_POINT)
					deformed_candidates.append(pc_src)
				except:
					got_error = True
					break					

			else:
				pc_src = OBJ_POINTCLOUDS[neighbors_idxs[i][j]]
			all_pc_src.append(pc_src)
			
		if (got_error):
			continue

		all_pc_src = np.array(all_pc_src)


		if (TO_DEFORM):
			deformed_candidates = np.array(deformed_candidates)
			deformed_pcs.append(deformed_candidates)

		feed_dict = {ops['pointclouds_pl_1']: all_pc_ref,
		         ops['pointclouds_pl_2']: all_pc_src,}
		chamfer_distance = sess.run([ops['chamfer_distance']], feed_dict=feed_dict)
		chamfer_distance = np.array(chamfer_distance)[0]
		cd_errors.append(chamfer_distance)

		if (i%50==0):
			print("Time elapsed: "+str(time.time()-start_time)+" sec for "+str(i)+"/"+str(num_samples)+" samples.")

	if (TO_DEFORM):
		deformed_pcs = np.array(deformed_pcs)
		print("Deformed pcs shape:")
		print(deformed_pcs.shape)

		filename = os.path.join(DUMP_DIR, 'deformation_parallel_newcost', 'deformed_retrievals_pcs.pickle')
		with open(filename, 'w') as handle:
			pickle.dump(deformed_pcs, handle, protocol=pickle.HIGHEST_PROTOCOL)

	cd_errors = np.array(cd_errors)
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

