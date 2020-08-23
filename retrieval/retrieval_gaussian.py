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
import provider
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from pdb import set_trace

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_gaussian', help='Model name [default: model]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--num_neighbors', type=int, default=3, help='Number of neighbors to retrieve')

parser.add_argument('--model_path', default='log_chair_ours_margin/model_0.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump_chair_ours_margin/', help='dump folder path [dump]')

parser.add_argument('--category', default='chair', help='Which class')

parser.add_argument('--output_dim', type=int, default=256, help='with or without autoencoder for triplet')
parser.add_argument('--use_mahalanobis', type=bool, default=True, help='Whether to use mahalanobis distance or euclidean')

parser.add_argument('--testrank', default=False, help='if testing with a smaller database for each model')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
OBJ_CAT = FLAGS.category

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model) # import network module

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

TRAIN_FILE = '../candidate_generation/train_'+OBJ_CAT+'.h5'
TEST_FILE = '../candidate_generation/test_'+OBJ_CAT+'.h5'

TRAIN_DATA = provider.load_h5(TRAIN_FILE)
TEST_DATA = provider.load_h5(TEST_FILE)

NUM_NEIGHBORS = FLAGS.num_neighbors
USE_MH = FLAGS.use_mahalanobis

OUTPUT_DIM = FLAGS.output_dim

TEST_RANK = FLAGS.testrank
if (TEST_RANK):
    pickle_in = open('../candidate_generation/candidates_test_'+OBJ_CAT+'_testrank.pickle',"rb")
    database_candidate_idxs = pickle.load(pickle_in)
    pickle_in.close()
    NUM_CANDIDATES = len(database_candidate_idxs[0]) 

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

            pointclouds_pl= MODEL.placeholder_inputs(BATCH_SIZE, 1, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            with tf.variable_scope("query_triplets") as scope:
                out_vecs, features= MODEL.get_model(pointclouds_pl, is_training_pl)
                mus, sigmas = MODEL.gaussian_params(out_vecs, output_dim = OUTPUT_DIM)

            # For inference
            qvec_pl = tf.placeholder(tf.float32, shape=(1, 1, OUTPUT_DIM))
            mus_pl = tf.placeholder(tf.float32, shape=(1, BATCH_SIZE, OUTPUT_DIM))
            sigmas_pl = tf.placeholder(tf.float32, shape=(1, BATCH_SIZE, OUTPUT_DIM))

            # _, probs = MODEL.compute_gaussian_probs(qvec_pl, mus_pl, sigmas_pl)
            if (USE_MH):
                probs = MODEL.compute_mahalanobis(qvec_pl, mus_pl, sigmas_pl)
            else:
                probs = MODEL.compute_euclidean(qvec_pl, mus_pl)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'qvec_pl': qvec_pl,
               'mus_pl': mus_pl,
               'sigmas_pl': sigmas_pl,
               'probs': probs,
               'mus': mus,
               'sigmas': sigmas}

        eval_one_epoch(sess, ops)

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    current_data = provider.get_current_data(TEST_DATA, NUM_POINT, shuffle=False)
    num_batches = current_data.shape[0]//BATCH_SIZE
    
    log_string(str(datetime.now()))
    log_string(str(current_data.shape[0]))

    loss_sum = 0

    all_mus = []
    all_sigmas = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_data= np.expand_dims(batch_data,axis=1)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['is_training_pl']: is_training}
        mus, sigmas = sess.run([ops['mus'], ops['sigmas']], feed_dict=feed_dict)
        all_mus.append(mus)
        all_sigmas.append(sigmas)

    all_mus = np.array(all_mus)
    all_mus = all_mus.reshape(-1, all_mus.shape[-1])
    all_sigmas = np.array(all_sigmas)
    all_sigmas = all_sigmas.reshape(-1, all_sigmas.shape[-1])

    #leftovers
    for i in range((current_data.shape[0]//BATCH_SIZE*BATCH_SIZE), current_data.shape[0]):
        pc = current_data[i,:,:]
        pc = np.expand_dims(pc, axis=0)
        fake_pcs = np.zeros((BATCH_SIZE-1, NUM_POINT,3))
        q = np.vstack((pc,fake_pcs))
        q = np.expand_dims(q,axis=1)

        feed_dict = {ops['pointclouds_pl']: q,
                     ops['is_training_pl']: is_training}
        mus, sigmas = sess.run([ops['mus'], ops['sigmas']], feed_dict=feed_dict)

        mus = mus[0][0]
        mus = np.squeeze(mus)
        sigmas = sigmas[0][0]
        sigmas = np.squeeze(sigmas)

        all_mus = np.vstack((all_mus, mus))
        all_sigmas = np.vstack((all_sigmas, sigmas))

    log_string(str(all_mus.shape[0]))
    log_string(str(all_sigmas.shape[0]))

    print(all_mus)
    print(all_sigmas)

    print("")

    ### Compute for probs
    neighbor_list = []
    for j in range(current_data.shape[0]):
        # print(j)

        q_vec = all_mus[j,:]
        q_vec = np.expand_dims(q_vec, axis=0)        
        q_vec = np.expand_dims(q_vec, axis=0)        

        all_probs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            batch_mus = all_mus[start_idx:end_idx, :]
            batch_mus= np.expand_dims(batch_mus,axis=0)
            batch_sigmas = all_sigmas[start_idx:end_idx, :]
            batch_sigmas= np.expand_dims(batch_sigmas,axis=0)

            feed_dict = { ops['qvec_pl']: q_vec,
                        ops['mus_pl']: batch_mus,
                        ops['sigmas_pl']: batch_sigmas}

            probs = sess.run([ops['probs']], feed_dict=feed_dict)
            all_probs.append(probs)

        all_probs = np.array(all_probs)
        all_probs = all_probs.reshape(-1, 1)

        #leftovers
        for i in range((current_data.shape[0]//BATCH_SIZE*BATCH_SIZE), current_data.shape[0]):
            batch_mus = all_mus[i, :]
            batch_sigmas = all_sigmas[i, :]

            fake_mus = np.zeros((BATCH_SIZE-1, OUTPUT_DIM))
            mus_stacked = np.vstack((batch_mus,fake_mus))
            fake_sigmas = np.zeros((BATCH_SIZE-1, OUTPUT_DIM))
            sigmas_stacked = np.vstack((batch_sigmas,fake_sigmas))

            mus_stacked = np.expand_dims(mus_stacked,axis=0)
            sigmas_stacked = np.expand_dims(sigmas_stacked,axis=0) 

            feed_dict = { ops['qvec_pl']: q_vec,
                        ops['mus_pl']: mus_stacked,
                        ops['sigmas_pl']: sigmas_stacked}

            probs = sess.run([ops['probs']], feed_dict=feed_dict)

            prob = probs[0][0][0]
            prob = np.squeeze(prob)

            all_probs = np.vstack((all_probs, prob))

        all_probs = np.squeeze(all_probs)  ###probs are distances lower the better

        if not TEST_RANK:
            idx = np.argsort(all_probs)
            j_nbr_idx = idx[:NUM_NEIGHBORS+1]
            j_nbr_idx = np.delete(j_nbr_idx, np.where(j_nbr_idx==j))
            neighbor_list.append(j_nbr_idx)

        else:
            database_candidates_idx_j = database_candidate_idxs[j]
            all_probs_candidates = all_probs[np.array(database_candidates_idx_j)]

            idx = np.argsort(all_probs_candidates)
            j_nbr_idx = idx[:NUM_NEIGHBORS]
            neighbor_list.append(j_nbr_idx)


    print(neighbor_list)

    pickle_out = open(os.path.join(DUMP_DIR, "neighbors.pickle"),"wb")
    pickle.dump(neighbor_list, pickle_out)
    pickle_out.close()

    log_string("Done.")    
         
    return 


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()

