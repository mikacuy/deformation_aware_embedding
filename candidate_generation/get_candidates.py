import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import os.path
from os import path
import json
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '..'))

sys.path.append(os.path.join(BASE_DIR, 'tf_ops/nn_distance'))
import tf_nndistance

import shutil
import time

import mesh_utils
import objloader

SHAPENET_BASEDIR = '/orion/group/ShapeNetManifold_10000/'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--category', default="chair", help='Which single class to use')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--num_candidates', type=int, default=50, help='Number of \'nearest neighbors\' to take.')
parser.add_argument('--generate_negatives', default = False, help='for generating negatives [default: False]')
parser.add_argument('--data_split', default = "train", help='which data split to use')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
OBJ_CAT = FLAGS.category
NUM_CANDIDATES = FLAGS.num_candidates
GENERATE_NEGS = FLAGS.generate_negatives
DATA_SPLIT = FLAGS.data_split

HOSTNAME = socket.gethostname()
np.random.seed(0)

#### Get point clouds for each object instance
with open('../shapenetcore_v2_split.json') as json_file:
    data = json.load(json_file)

train_data = data[DATA_SPLIT]
num_categories = len(list(train_data.keys()))

idx = -1
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

#####h5 file handles
def save_dataset(fname, pcs):
    cloud = np.stack([pc for pc in pcs])

    fout = h5py.File(fname)
    fout.create_dataset('data', data=cloud, compression='gzip', dtype='float32')
    fout.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    return data

# ##load obj files and get point clouds
start_time = time.time()

fname = DATA_SPLIT +"_"+OBJ_CAT+".h5"

if not path.exists(fname):
    OBJ_POINTCLOUDS = []
    for i in range(len(model_names)):
        ref_model_name = model_names[i]

        ref_filename = os.path.join(SHAPENET_BASEDIR, synsetid, ref_model_name, 'models', 'model_normalized.obj')
        V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(ref_filename)

        idx_pts = np.arange(V.shape[0])
        np.random.shuffle(idx_pts)

        if(V.shape[0]<NUM_POINT):
            pass

        pc = mesh_utils.mesh_to_pc(V, F, num_points=NUM_POINT)

        OBJ_POINTCLOUDS.append(pc)

        if (i%50==0):
            print("Time elapsed: "+str(time.time()-start_time)+" sec for "+str(i)+" samples.")

    OBJ_POINTCLOUDS = np.array(OBJ_POINTCLOUDS)
    save_dataset(fname, OBJ_POINTCLOUDS)

OBJ_POINTCLOUDS = load_h5(fname)
print(OBJ_POINTCLOUDS.shape)
print("Loading and sampling time: "+str(time.time()-start_time)+" sec")
print("Done processing h5 files.")

def chamfer_loss(pc1, pc2):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pc1, pc2)
    # loss = dists_forward+dists_backward
    loss = tf.reduce_mean(dists_forward+dists_backward, axis=1)
    return loss

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl_1 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
            pointclouds_pl_2 = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))

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

        epoch_loss = eval_one_epoch(sess, ops)        

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """

    current_data = OBJ_POINTCLOUDS
    num_batches = current_data.shape[0]//BATCH_SIZE

    candidates_idx = []
    start_time = time.time()
    for j in range(current_data.shape[0]):
        curr_pc = current_data[j, :, :]
        curr_pc = np.expand_dims(curr_pc, axis=0)
        curr_pc = np.tile(curr_pc, (BATCH_SIZE, 1, 1))

        #To compare
        chamfer_distances = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            batch_data = current_data[start_idx:end_idx, :, :]

            feed_dict = {ops['pointclouds_pl_1']: curr_pc,
                         ops['pointclouds_pl_2']: batch_data,}
            chamfer_distance = sess.run([ops['chamfer_distance']], feed_dict=feed_dict)
            chamfer_distance = np.array(chamfer_distance)[0]

            for i in range(BATCH_SIZE):
                chamfer_distances.append(chamfer_distance[i])

        for extra in range((num_batches)*BATCH_SIZE, current_data.shape[0]):
            extra_pc = current_data[extra, :, :]
            extra_pc = np.expand_dims(extra_pc, axis=0)
            extra_pc = np.tile(extra_pc, (BATCH_SIZE, 1, 1))

            feed_dict = {ops['pointclouds_pl_1']: curr_pc,
                         ops['pointclouds_pl_2']: extra_pc,}
            chamfer_distance = sess.run([ops['chamfer_distance']], feed_dict=feed_dict)
            chamfer_distance = np.array(chamfer_distance)[0]

            #Only the first pair is not a placeholder 
            chamfer_distances.append(chamfer_distance[0])   

        chamfer_distances = np.array(chamfer_distances)

        if (GENERATE_NEGS):
            #Get random 25 from (1/3, 2/3) * num_samples and 25 from (2/3, 1) * num_samples
            num_samples = chamfer_distances.shape[0]
            idx = np.argsort(chamfer_distances)

            med_cd_idx = idx[int(num_samples/3):2*int(num_samples/3)]
            m_idx_select = np.random.choice(med_cd_idx.shape[0], int(NUM_CANDIDATES/2), replace=False)
            med_cd_idx_selected = med_cd_idx[m_idx_select]

            far_cd_idx = idx[2*int(num_samples/3):]
            f_idx_select = np.random.choice(far_cd_idx.shape[0], int(NUM_CANDIDATES/2), replace=False)
            far_cd_idx_selected = far_cd_idx[f_idx_select]

            selected = np.concatenate((med_cd_idx_selected, far_cd_idx_selected), axis=0)
            candidates_idx.append(selected)

        ###Gets top 50 chamfer distance candidates
        else: 
            idx = np.argsort(chamfer_distances) ## sorted but slower runtime

            #Remove itself
            idx_candidates = np.delete(idx, np.argwhere(idx==j))
            candidates_idx.append(idx_candidates[:NUM_CANDIDATES])

        if (j%100==0):
            print("Time elapsed: "+str(time.time()-start_time)+" sec for "+str(j)+" samples.")

    print(candidates_idx[0])
    print(candidates_idx[100])

    if (GENERATE_NEGS):
        filename = 'candidates_' + DATA_SPLIT + '_'+OBJ_CAT+'_negatives.pickle'

    else:
        filename = 'candidates_' + DATA_SPLIT + '_'+OBJ_CAT+'_retrieval.pickle'

    with open(filename, 'w') as handle:
        pickle.dump(candidates_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(filename)
    print("Done.")

if __name__ == "__main__":
    evaluate()

