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

parser.add_argument('--dump_dir', default='dump_chair_ranked_cd/', help='dump folder path [dump]')

parser.add_argument('--fitting_dump_dir', default='test_rank_point2mesh/', help='dump folder path after fitting')
# parser.add_argument('--fitting_dump_dir', default='deformation_parallel_newcost_2cd/', help='dump folder path after fitting')
parser.add_argument('--to_deform', default=True, help='with or without deformation')
parser.add_argument('--num_neighbors', type=int, default=3, help='Number of neighbors to retrieve')


FLAGS = parser.parse_args() 
OBJ_CAT = FLAGS.category
DATA_SPLIT = FLAGS.data_split
NUM_NEIGHBORS = FLAGS.num_neighbors
DUMP_DIR = str(FLAGS.dump_dir)
print(DUMP_DIR)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

FITTING_DUMP_DIR = os.path.join(DUMP_DIR, FLAGS.fitting_dump_dir)
if not os.path.exists(FITTING_DUMP_DIR): os.mkdir(FITTING_DUMP_DIR)
LOG_FOUT = open(os.path.join(FITTING_DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

TO_DEFORM = FLAGS.to_deform
print("Deform "+str(TO_DEFORM))

# if TO_DEFORM:
# 	print("ERROR. Please run evaluate_fitting_deform.py instead.")
# 	exit()

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

# #Retrieved neighbor indices
# pickle_in = open(os.path.join(DUMP_DIR, "neighbors.pickle"),"rb")
# neighbors_idxs = pickle.load(pickle_in)

shapes = data[str(cat_idx)]
synsetid = shapes["synsetid"]
model_names = shapes["model_names"]
num_samples = shapes["num_samples"]
NUM_POINT = 2048

####Get candidates
pickle_in = open('../candidate_generation/candidates_'+DATA_SPLIT+'_'+OBJ_CAT+'_testrank.pickle',"rb")
database_candidate_idxs = pickle.load(pickle_in)
pickle_in.close()
NUM_CANDIDATES = len(database_candidate_idxs[0])

####Get pre-computed deformed chamfer distance
FOL = "chamfer_distance_deformed_candidates/"
pickle_in = open(os.path.join(FOL, "testrank_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+"_point2mesh.pickle"))
database_deformedCD_costs = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join(FOL, "testrank_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+"_point2mesh_undeformed.pickle"))
database_CD_costs = pickle.load(pickle_in)
pickle_in.close()


def chamfer_loss(pc1, pc2):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pc1, pc2)
    # loss = dists_forward+dists_backward
    loss = tf.reduce_mean(dists_forward+dists_backward, axis=1)
    return loss

with tf.Graph().as_default():
	with tf.device('/gpu:0'):
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

	# fname = DATA_SPLIT +"_"+OBJ_CAT+"_meshsampled.h5"
	fname = '../candidate_generation/' + DATA_SPLIT +"_"+OBJ_CAT + '.h5'
	OBJ_POINTCLOUDS = load_h5(fname)
	print(OBJ_POINTCLOUDS.shape)


	all_cd = []
	all_deformed_cd = []
	all_cd_ranks = []
	all_deformed_cd_ranks = []
	for i in range(len(model_names)):
		# pc_ref = OBJ_POINTCLOUDS[i]
		# all_pc_ref = []
		# for _ in range(NUM_CANDIDATES):
		# 	all_pc_ref.append(pc_ref)
		# all_pc_ref = np.array(all_pc_ref)

		# database_candidates_idx_i = database_candidate_idxs[i]
		# all_pc_src = []
		# for j in range(NUM_CANDIDATES):
		# 	pc_src = OBJ_POINTCLOUDS[database_candidates_idx_i[j]]
		# 	all_pc_src.append(pc_src)

		# all_pc_src = np.array(all_pc_src)

		# ##CD before deformation
		# feed_dict = {ops['pointclouds_pl_1']: all_pc_ref,
		#          ops['pointclouds_pl_2']: all_pc_src,}
		# chamfer_distances = sess.run([ops['chamfer_distance']], feed_dict=feed_dict)
		# chamfer_distances = np.array(chamfer_distances)[0]

		chamfer_distances = database_CD_costs[i]

		chamfer_distance_idx_sorted = np.argsort(chamfer_distances)
		retrieved_neighbors_idx = chamfer_distance_idx_sorted[:NUM_NEIGHBORS]


		##Deformed CD
		deformed_chamfer_distances = database_deformedCD_costs[i]
		deformed_chamfer_distances[np.argwhere(deformed_chamfer_distances==-1)] = 1e16 #those with invalid deformation


		retrieved_chamfer_distances = chamfer_distances[retrieved_neighbors_idx]
		retrieved_deformed_chamfer_distances = deformed_chamfer_distances[retrieved_neighbors_idx]

		if (np.max(retrieved_deformed_chamfer_distances) > 1000):
			continue
		
		deformed_chamfer_distance_idx_sorted = np.argsort(deformed_chamfer_distances)

		CD_ranks = np.empty_like(chamfer_distance_idx_sorted)
		CD_ranks[chamfer_distance_idx_sorted] = np.arange(NUM_CANDIDATES)
		deformed_CD_ranks = np.empty_like(deformed_chamfer_distance_idx_sorted)
		deformed_CD_ranks[deformed_chamfer_distance_idx_sorted] = np.arange(NUM_CANDIDATES)

		retrieved_CD_rank = CD_ranks[retrieved_neighbors_idx] + 1 
		retrieved_deformed_CD_rank = deformed_CD_ranks[retrieved_neighbors_idx] + 1


		all_cd.append(retrieved_chamfer_distances)
		all_deformed_cd.append(retrieved_deformed_chamfer_distances)
		all_cd_ranks.append(retrieved_CD_rank)
		all_deformed_cd_ranks.append(retrieved_deformed_CD_rank)

	all_cd = np.array(all_cd)
	all_deformed_cd = np.array(all_deformed_cd)
	all_cd_ranks = np.array(all_cd_ranks)
	all_deformed_cd_ranks = np.array(all_deformed_cd_ranks)

	for i in range(NUM_NEIGHBORS):
		i_mean_cd = np.mean(all_cd[:,i])
		i_mean_cd_rank = np.mean(all_cd_ranks[:,i])
		i_mean_deformed_cd = np.mean(all_deformed_cd[:,i])
		i_mean_deformed_cd_rank = np.mean(all_deformed_cd_ranks[:,i])
		log_string("Rank "+ str(i+1) + " retrieved mean CD error: "+str(i_mean_cd))
		log_string("Rank "+ str(i+1) + " retrieved mean CD rank: "+str(i_mean_cd_rank))
		log_string("Rank "+ str(i+1) + " retrieved mean deformed CD error: "+str(i_mean_deformed_cd))
		log_string("Rank "+ str(i+1) + " retrieved mean deformed CD rank: "+str(i_mean_deformed_cd_rank))
		log_string(" ")

	# log_string(" ")
	# log_string("Recall")
	# log_string("K= "+str(K))
	# log_string(" ")
	# for i in range(NUM_NEIGHBORS):
	# 	log_string("Recall@"+ str(i+1) + ": "+str(recall[i]))


	# log_string(" ")
	print("Total running time: "+str(time.time()-start_time)+" sec")
	LOG_FOUT.close()


