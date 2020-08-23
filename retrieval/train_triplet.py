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
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import time
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='chair', help='Which class')

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_triplet', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_chair_triplet/', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=401, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

####For triplets
parser.add_argument('--margin', type=float, default=0.5, help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--positives_per_query', type=int, default=2, help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=13, help='Number of definite negatives in each training tuple [default: 18]')

parser.add_argument('--output_dim', type=int, default=256, help='with or without autoencoder for triplet')

FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

##For triplet
POSITIVES_PER_QUERY= FLAGS.positives_per_query
NEGATIVES_PER_QUERY= FLAGS.negatives_per_query
MARGIN = FLAGS.margin

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(os.path.join(ROOT_DIR, 'models'), FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_triplet.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

OBJ_CAT = FLAGS.category
TRAIN_FILE = '../candidate_generation/train_'+OBJ_CAT+'.h5'
TEST_FILE = '../candidate_generation/test_'+OBJ_CAT+'.h5'
TRAIN_DATA = provider.load_h5(TRAIN_FILE)
TEST_DATA = provider.load_h5(TEST_FILE)

TRAIN_CANDIDATES_FILE = 'generate_deformed_candidates/chamfer_triplet_train_'+OBJ_CAT+'.pickle'
TEST_CANDIDATES_FILE = 'generate_deformed_candidates/chamfer_triplet_test_'+OBJ_CAT+'.pickle'

pickle_in = open(TRAIN_CANDIDATES_FILE,"rb")
TRAIN_DICT = pickle.load(pickle_in)
pickle_in = open(TEST_CANDIDATES_FILE,"rb")
TEST_DICT = pickle.load(pickle_in)

OUTPUT_DIM = FLAGS.output_dim

np.random.seed(0)

global TRAINING_LATENT_VECTORS
TRAINING_LATENT_VECTORS=[]

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            query= MODEL.placeholder_inputs(BATCH_SIZE, 1, NUM_POINT)
            positives= MODEL.placeholder_inputs(BATCH_SIZE, POSITIVES_PER_QUERY, NUM_POINT)
            negatives= MODEL.placeholder_inputs(BATCH_SIZE, NEGATIVES_PER_QUERY, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print (is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print ("--- Get model and loss")
            with tf.variable_scope("query_triplets") as scope:
                vecs= tf.concat([query, positives, negatives],1)
                print(vecs)                
                out_vecs, end_points= MODEL.get_model(vecs, is_training_pl, autoencoder=False, bn_decay=bn_decay, output_dim=OUTPUT_DIM)
                print(out_vecs)
                q_vec, pos_vecs, neg_vecs= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)

            # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)

            # Get loss 
            loss = MODEL.triplet_loss(q_vec, pos_vecs, neg_vecs, MARGIN)
            # loss = MODEL.chamfer_loss(pred, pointclouds_pl)
            tf.summary.scalar('triplet_loss', loss)

            print ("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        #sess.run(init, {is_training_pl: True})

        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'q_vec':q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs,                   
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_loss = 1e20
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)

            ### Train with hard negatives
            # if (epoch > 50):
            #     train_one_epoch(sess, ops, train_writer, use_hard_neg=True, recache=True)

            # else:
            #     train_one_epoch(sess, ops, train_writer)

            epoch_loss = eval_one_epoch(sess, ops, test_writer)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer, use_hard_neg=False, recache = False):
    """ ops: dict mapping from string to tf ops """
    global TRAINING_LATENT_VECTORS    
    is_training = True
    
    # Sample and shuffle train samples
    current_data, positives, negatives, idx = provider.get_current_data_arap(TRAIN_DATA, TRAIN_DICT, POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, NUM_POINT, shuffle=True)


    log_string(str(datetime.now()))
    num_batches = current_data.shape[0]//BATCH_SIZE

    train_pcs = provider.get_current_data(TRAIN_DATA, NUM_POINT, shuffle=False)

    if (use_hard_neg and len(TRAINING_LATENT_VECTORS)==0):
        TRAINING_LATENT_VECTORS = get_latent_vectors(sess, ops)
    num_hard_negs = 8


    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data = current_data[start_idx:end_idx, :, :]
        batch_data= np.expand_dims(batch_data,axis=1)
        curr_positives = positives[start_idx:end_idx, :, :, :]

        if (use_hard_neg):
            # print("Using hard negatives")
            curr_idx = idx[start_idx:end_idx]
            # current_queries = train_pcs[curr_idx]
            # query_feat = get_feature_representation(current_queries, sess, ops)
            curr_queries_latent_vector = TRAINING_LATENT_VECTORS[curr_idx]

            curr_negatives = []
            for j in range(BATCH_SIZE):
                neg_idx = TRAIN_DICT["negatives"][curr_idx[j]] #returns a list

                # if (len(neg_idx) == 0):
                #     continue

                if (len(neg_idx) < NEGATIVES_PER_QUERY):
                    selected_idx = np.random.choice(neg_idx, NEGATIVES_PER_QUERY, replace=True)                    
                else:
                    neg_latent_vec = TRAINING_LATENT_VECTORS[np.array(neg_idx)]
                    query_vec = curr_queries_latent_vector[j]
                    hard_negs = get_hard_negatives(query_vec, neg_latent_vec, neg_idx, num_hard_negs)

                    ##Get negative pcs
                    if (len(neg_idx) - num_hard_negs < NEGATIVES_PER_QUERY):
                        selected_idx = np.random.choice(neg_idx, NEGATIVES_PER_QUERY, replace=False)
                        selected_idx[:num_hard_negs] = np.array(hard_negs)

                    else:
                        neg_idx = np.delete(np.array(neg_idx), np.where(np.isin(np.array(neg_idx) ,np.array(hard_negs))))

                        to_select_idx = np.arange(0,len(neg_idx))
                        np.random.shuffle(to_select_idx)
                        selected_idx = neg_idx[to_select_idx[0:NEGATIVES_PER_QUERY]]

                        selected_idx[:num_hard_negs] = np.array(hard_negs)

                curr_neg_pcs = train_pcs[selected_idx]
                curr_negatives.append(curr_neg_pcs)

            curr_negatives = np.array(curr_negatives)

            if (len(curr_negatives.shape) != 4 or curr_negatives.shape[0]!=BATCH_SIZE):
                continue

        else:
            curr_negatives = negatives[start_idx:end_idx, :, :, :]

        feed_dict = {ops['query']: batch_data,
                    ops['positives']: curr_positives,
                    ops['negatives']: curr_negatives,
                    ops['is_training_pl']: is_training}
        summary, step, _, loss_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            loss_sum = 0
        

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # current_data = provider.get_current_data(TEST_DATA, NUM_POINT)
    current_data, positives, negatives = provider.get_current_data_triplet(TEST_DATA, TEST_DICT, POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY, NUM_POINT, shuffle=True)
    num_batches = current_data.shape[0]//BATCH_SIZE
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_data= np.expand_dims(batch_data,axis=1)
        curr_positives = positives[start_idx:end_idx, :, :, :]
        curr_negatives = negatives[start_idx:end_idx, :, :, :]

        feed_dict = {ops['query']: batch_data,
                    ops['positives']: curr_positives,
                    ops['negatives']: curr_negatives,
                    ops['is_training_pl']: is_training}
        summary, step, loss_val = sess.run([ops['merged'], ops['step'],
            ops['loss']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        loss_sum += loss_val
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
         
    EPOCH_CNT += 1
    return loss_sum/float(num_batches)

def get_hard_negatives(query_vec, neg_latent_vecs, neg_idx, num_to_take):
    neg_latent_vecs=np.array(neg_latent_vecs)
    nbrs = KDTree(neg_latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]),k=num_to_take)
    hard_negs=np.squeeze(np.array(neg_idx)[indices[0]])
    hard_negs= hard_negs.tolist()
    return hard_negs

def get_latent_vectors(sess, ops):
    print("Caching latent features.")
    start_time = time.time()

    is_training=False
    train_idxs = np.arange(0, TRAIN_DATA.shape[0])
    train_pcs = provider.get_current_data(TRAIN_DATA, NUM_POINT, shuffle=False)

    batch_num= BATCH_SIZE*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(train_idxs.shape[0]//batch_num):

        queries = train_pcs[q_index*batch_num:(q_index+1)*(batch_num)]

        q1=queries[0:BATCH_SIZE]
        q1=np.expand_dims(q1,axis=1)

        q2=queries[BATCH_SIZE:BATCH_SIZE*(POSITIVES_PER_QUERY+1)]
        q2=np.reshape(q2,(BATCH_SIZE,POSITIVES_PER_QUERY,NUM_POINT,3))

        q3=queries[BATCH_SIZE*(POSITIVES_PER_QUERY+1):BATCH_SIZE*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        q3=np.reshape(q3,(BATCH_SIZE,NEGATIVES_PER_QUERY,NUM_POINT,3))


        feed_dict={ops['query']:q1, ops['positives']:q2, ops['negatives']:q3, ops['is_training_pl']:is_training}
        o1, o2, o3=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], feed_dict=feed_dict)
        
        o1=np.reshape(o1,(-1,o1.shape[-1]))
        o2=np.reshape(o2,(-1,o2.shape[-1]))
        o3=np.reshape(o3,(-1,o3.shape[-1]))        

        out=np.vstack((o1,o2,o3))
        q_output.append(out)

    q_output=np.array(q_output)
    if(len(q_output)!=0):  
        q_output=q_output.reshape(-1,q_output.shape[-1])

    #handle edge case
    for q_index in range((train_idxs.shape[0]//batch_num*batch_num),train_idxs.shape[0]):
        queries = train_pcs[q_index]
        queries= np.expand_dims(queries,axis=0)
        queries= np.expand_dims(queries,axis=0)
        # print(queries.shape)

        if(BATCH_SIZE-1>0):
            fake_queries=np.zeros((BATCH_SIZE-1,1,NUM_POINT,3))
            # print(fake_queries.shape)
            q=np.vstack((queries,fake_queries))
        else:
            q=queries

        fake_pos=np.zeros((BATCH_SIZE,POSITIVES_PER_QUERY,NUM_POINT,3))
        fake_neg=np.zeros((BATCH_SIZE,NEGATIVES_PER_QUERY,NUM_POINT,3))
        feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['is_training_pl']:is_training}
        output=sess.run(ops['q_vec'], feed_dict=feed_dict)
        output=output[0]
        output=np.squeeze(output)
        if (q_output.shape[0]!=0):
            q_output=np.vstack((q_output,output))
        else:
            q_output=output

    # print(q_output.shape)
    print("Time elapsed: "+str(time.time()-start_time)+" sec for caching latent vectors.")
    return q_output


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()

