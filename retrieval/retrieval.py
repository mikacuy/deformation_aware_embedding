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
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_autoencoder', help='Model name [default: model]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--num_neighbors', type=int, default=3, help='Number of neighbors to retrieve')

parser.add_argument('--model_path', default='log_chair_autoencoder/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump_chair_autoencoder/', help='dump folder path [dump]')
parser.add_argument('--category', default='chair', help='Which class')
parser.add_argument('--output_dim', type=int, default=256, help='with or without autoencoder for triplet')

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

            if (FLAGS.model == "pointnet_autoencoder"):
                pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
                is_training_pl = tf.placeholder(tf.bool, shape=())

                print ("--- Get model and loss")
                # Get model and loss 
                pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
                embeddings = end_points['embedding']

            elif FLAGS.model == "pointnet_autoencoder_dimreduc":
                pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
                is_training_pl = tf.placeholder(tf.bool, shape=())                
                pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, output_dim = OUTPUT_DIM)
                embeddings = end_points['embedding']

            else:
                pointclouds_pl= MODEL.placeholder_inputs(BATCH_SIZE, 1, NUM_POINT)
                is_training_pl = tf.placeholder(tf.bool, shape=())
                with tf.variable_scope("query_triplets") as scope:
                    if FLAGS.model == "pointnet_triplet":
                        out_vecs, end_points= MODEL.get_model(pointclouds_pl, is_training_pl, autoencoder=False, output_dim=OUTPUT_DIM)

                    else:
                        out_vecs, _, end_points= MODEL.get_model(pointclouds_pl, is_training_pl, autoencoder=True)
                out_vecs = tf.squeeze(out_vecs)
                embeddings = out_vecs
                pred = None #dummy
            
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
               'pred': pred,
               'embeddings': embeddings,
               'end_points': end_points}

        eval_one_epoch(sess, ops)

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    current_data = provider.get_current_data(TEST_DATA, NUM_POINT, shuffle=False)
    num_batches = current_data.shape[0]//BATCH_SIZE
    
    log_string(str(datetime.now()))
    log_string(str(current_data.shape[0]))

    loss_sum = 0

    all_embeddings = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data = current_data[start_idx:end_idx, :, :]

        if (FLAGS.model != "pointnet_autoencoder" and FLAGS.model != "pointnet_autoencoder_dimreduc"):
            batch_data= np.expand_dims(batch_data,axis=1)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['is_training_pl']: is_training}
        embeddings = sess.run([ops['embeddings']], feed_dict=feed_dict)
        all_embeddings.append(embeddings)

    all_embeddings = np.array(all_embeddings)
    all_embeddings = all_embeddings.reshape(-1,all_embeddings.shape[-1])

    #leftovers
    for i in range((current_data.shape[0]//BATCH_SIZE*BATCH_SIZE), current_data.shape[0]):
        pc = current_data[i,:,:]
        pc= np.expand_dims(pc, axis=0)
        fake_pcs = np.zeros((BATCH_SIZE-1, NUM_POINT,3))
        q=np.vstack((pc,fake_pcs))

        if (FLAGS.model != "pointnet_autoencoder" and FLAGS.model != "pointnet_autoencoder_dimreduc"):
            q= np.expand_dims(q,axis=1)

        feed_dict = {ops['pointclouds_pl']: q,
                     ops['is_training_pl']: is_training}
        embeddings = sess.run([ops['embeddings']], feed_dict=feed_dict)

        embeddings = embeddings[0][0]
        embeddings=np.squeeze(embeddings)
        all_embeddings = np.vstack((all_embeddings, embeddings))

    log_string(str(all_embeddings.shape[0]))

    # tsne = TSNE(n_components=2, random_state=0)
    # embeddings_2d = tsne.fit_transform(all_embeddings)

    # for i in range(len(embeddings)):
    #     plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])

    # plt.savefig(os.path.join(DUMP_DIR, 'tsne.png'))

    ####For Retrieval
    if not TEST_RANK:
        database_kdtree = KDTree(all_embeddings)

        neighbor_list = []
        distances, nbr_idx = database_kdtree.query(all_embeddings, k=NUM_NEIGHBORS+1)
        nbr_idx = np.squeeze(nbr_idx)
        for i in range(all_embeddings.shape[0]):
            i_nbr_idx = np.delete(nbr_idx[i], np.where(nbr_idx[i]==i))
            i_nbr_idx = i_nbr_idx[:NUM_NEIGHBORS]

            neighbor_list.append(i_nbr_idx)

    ###For smaller subset of the database
    else:
        neighbor_list = []
        for i in range(all_embeddings.shape[0]):
            database_candidates_idx_i = database_candidate_idxs[i]
            database_candidates_embeddings = all_embeddings[np.array(database_candidates_idx_i)]
            # print(database_candidates_embeddings.shape)
            # exit()

            database_kdtree_i = KDTree(database_candidates_embeddings)
            distances, nbr_idx = database_kdtree_i.query(np.array([all_embeddings[i]]), k=NUM_NEIGHBORS)
            # neighbor_list.append(database_candidates_idx_i[nbr_idx[0]])
            # print(database_candidates_idx_i[nbr_idx[0]])
            neighbor_list.append(nbr_idx[0])
            # print(nbr_idx[0])            
            # exit()


    pickle_out = open(os.path.join(DUMP_DIR, "neighbors.pickle"),"wb")
    pickle.dump(neighbor_list, pickle_out)
    pickle_out.close()

    log_string("Done.")    
         
    return 


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()

