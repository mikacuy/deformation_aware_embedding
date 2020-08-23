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

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='chair', help='Which class')

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_gaussian', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_chair_ours_reg/', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=301, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

parser.add_argument('--use_obj_sigma', type=int, default=1, help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--sigma', type=float, default=0.005, help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--candidates_per_query', type=int, default=15, help='Number of potential candidates in each training tuple [default: 15]')
parser.add_argument('--use_mahalanobis', type=bool, default=True, help='Whether to use mahalanobis distance or euclidean')

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

USE_OBJ_SIGMA = FLAGS.use_obj_sigma
USE_MH = FLAGS.use_mahalanobis

##For distance loss
CANDIDATES_PER_QUERY = FLAGS.candidates_per_query
SIGMA = FLAGS.sigma

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(os.path.join(ROOT_DIR, 'models'), FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_ours_distances.py %s' % (LOG_DIR)) # bkp of train procedure
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

TRAIN_CANDIDATES_FILE = 'generate_deformed_candidates/arap_distances_sigmas_train_'+OBJ_CAT+'.pickle'
pickle_in = open(TRAIN_CANDIDATES_FILE,"rb")
TRAIN_DICT = pickle.load(pickle_in)

# TEST_CANDIDATES_FILE = FLAGS.test_candidates_file
# pickle_in = open(TEST_CANDIDATES_FILE,"rb")
# TEST_DICT = pickle.load(pickle_in)

OUTPUT_DIM = FLAGS.output_dim

np.random.seed(0)

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
            candidates= MODEL.placeholder_inputs(BATCH_SIZE, CANDIDATES_PER_QUERY, NUM_POINT)
            costs= tf.placeholder(tf.float32, shape = (BATCH_SIZE, CANDIDATES_PER_QUERY))
            obj_sigmas = tf.placeholder(tf.float32, shape = (BATCH_SIZE))

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print (is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print ("--- Get model and loss")
            with tf.variable_scope("query_triplets") as scope:
                vecs= tf.concat([query, candidates],1)
                print(vecs)                
                out_vecs, features= MODEL.get_model(vecs, is_training_pl, autoencoder=False, bn_decay=bn_decay)
                print(out_vecs)

                mus, sigmas = MODEL.gaussian_params(out_vecs, output_dim = OUTPUT_DIM)

                q_mus, candidate_mus = tf.split(mus, [1,CANDIDATES_PER_QUERY],1)
                q_sigmas, candidate_sigmas = tf.split(sigmas, [1,CANDIDATES_PER_QUERY],1)         

            # Get loss 
            if (USE_OBJ_SIGMA == 1):
                print("Using per object sigma.")
                loss = MODEL.distance_regression_loss(q_mus, candidate_mus, candidate_sigmas, costs, obj_sigmas, use_mahalanobis = USE_MH)  
                # loss = MODEL.distance_regression_loss(q_mus, candidate_mus, candidate_sigmas, costs, obj_sigmas, use_l2 = True)
                # loss = MODEL.distance_KL_loss(q_mus, candidate_mus, candidate_sigmas, costs, obj_sigmas)
            else:
                loss = MODEL.distance_loss(q_mus, candidate_mus, candidate_sigmas, costs, SIGMA)

            tf.summary.scalar('loss', loss)

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
            saver = tf.train.Saver(max_to_keep=10)
        
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

        # # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ##Load pretrained
        # MODEL_PATH = "log_tables_chamfer_mahalanobis3/best_model_epoch_006.ckpt"
        # saver.restore(sess, MODEL_PATH)
        # log_string("Model restored.")

        ops = {'query': query,
               'candidates': candidates,
               'costs': costs,
               'mus':mus,
               'sigmas':sigmas,
               'obj_sigmas': obj_sigmas,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_loss = 1e20
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)

            # epoch_loss = eval_one_epoch(sess, ops, test_writer)
            # if epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
            #     log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_"+str(epoch)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Sample and shuffle train samples
    if USE_OBJ_SIGMA == 1:
        current_data, candidates_pcs, candidates_costs, current_sigmas, idx = provider.get_current_data_arap_distances_with_sigmas(TRAIN_DATA, TRAIN_DICT, CANDIDATES_PER_QUERY, NUM_POINT, shuffle=True)
    else:
        current_data, candidates_pcs, candidates_costs, idx = provider.get_current_data_arap_distances(TRAIN_DATA, TRAIN_DICT, CANDIDATES_PER_QUERY, NUM_POINT, shuffle=True)

    log_string(str(datetime.now()))
    num_batches = current_data.shape[0]//BATCH_SIZE

    total_loss_sum = 0
    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_data= np.expand_dims(batch_data,axis=1)
        curr_candidates = candidates_pcs[start_idx:end_idx, :, :, :]
        curr_costs = candidates_costs[start_idx:end_idx, :]

        feed_dict = {ops['query']: batch_data,
                    ops['candidates']: curr_candidates,
                    ops['costs']: curr_costs,
                    ops['is_training_pl']: is_training}

        if (USE_OBJ_SIGMA == 1):
            batch_sigmas = current_sigmas[start_idx:end_idx]
            feed_dict[ops["obj_sigmas"]] = batch_sigmas

        summary, step, _, loss_val, mus, sigmas = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['mus'], ops['sigmas']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            print("")

            total_loss_sum += loss_sum
            loss_sum = 0


    total_loss_sum /= num_batches
    return total_loss_sum
        

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    if USE_OBJ_SIGMA == 1:
        current_data, candidates_pcs, candidates_costs, current_sigmas, idx = provider.get_current_data_arap_distances_with_sigmas(TEST_DATA, TEST_DICT, CANDIDATES_PER_QUERY, NUM_POINT, shuffle=True)
    else:
        current_data, candidates_pcs, candidates_costs, idx = provider.get_current_data_arap_distances(TEST_DATA, TEST_DICT, CANDIDATES_PER_QUERY, NUM_POINT, shuffle=True)
    num_batches = current_data.shape[0]//BATCH_SIZE
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data = current_data[start_idx:end_idx, :, :]
        batch_data= np.expand_dims(batch_data,axis=1)
        curr_candidates = candidates_pcs[start_idx:end_idx, :, :, :]
        curr_costs = candidates_costs[start_idx:end_idx, :]

        feed_dict = {ops['query']: batch_data,
                    ops['candidates']: curr_candidates,
                    ops['costs']: curr_costs,
                    ops['is_training_pl']: is_training}

        if (USE_OBJ_SIGMA == 1):
            batch_sigmas = current_sigmas[start_idx:end_idx]
            feed_dict[ops["obj_sigmas"]] = batch_sigmas

        summary, step, loss_val = sess.run([ops['merged'], ops['step'],
            ops['loss']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        loss_sum += loss_val
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
         
    EPOCH_CNT += 1
    return loss_sum/float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()

