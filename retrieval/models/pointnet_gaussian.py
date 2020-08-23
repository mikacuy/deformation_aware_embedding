""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.
Using GPU Chamfer's distance loss.

Author: Charles R. Qi
Date: May 2018
"""
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))
import tf_nndistance

def placeholder_inputs(batch_num_queries, num_pointclouds_per_query, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_num_queries, num_pointclouds_per_query, num_point, 3))
    return pointclouds_pl

def get_model(point_cloud, is_training, autoencoder=False, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    batch_size = point_cloud.get_shape()[0].value
    num_pointclouds_per_query = point_cloud.get_shape()[1].value
    num_point = point_cloud.get_shape()[2].value
    point_cloud = tf.reshape(point_cloud, [batch_size*num_pointclouds_per_query, num_point,3])

    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    # Encoder
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    net = tf.reshape(global_feat, [batch_size*num_pointclouds_per_query, -1])
    feature = net

    net = tf_util.fully_connected(net, 512, bn=True,
            is_training=is_training, bn_decay=bn_decay,
            scope='fc1')

    net = tf_util.fully_connected(net, 256, bn=True,
            is_training=is_training, bn_decay=bn_decay,
            scope='fc2')

    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
            scope='dp1')

    net =  tf.reshape(net,[batch_size,num_pointclouds_per_query,-1])

    ### net = (batch_size, num_pointclouds_per_query, -1)
    return net, feature


def gaussian_params(net, max_sigma = 0.5, output_dim = 256):
    ### net = (batch_size*num_pointclouds_per_query, -1)
    batch_size = net.get_shape()[0].value
    num_pointclouds_per_query = net.get_shape()[1].value

    net =  tf.reshape(net,[batch_size*num_pointclouds_per_query,-1])
    mus = tf_util.fully_connected(net, output_dim,
            activation_fn=None, scope='mus')
    mus = tf.reshape(mus, [batch_size, num_pointclouds_per_query, output_dim])

    sigmas = tf_util.fully_connected(net, output_dim,
            activation_fn=None, scope='sigmas')
    sigmas = tf.sigmoid(sigmas) + 1.0e-6

    # sigmas = tf.exp(sigmas)
    # Clip sigmas.
    # max_sigma = 0.05
    # sigmas = tf.clip_by_value(sigmas, 1.0e-6, max_sigma)

    sigmas = tf.reshape(sigmas, [batch_size, num_pointclouds_per_query, output_dim])

    return mus, sigmas


def compute_gaussian_probs(query_vec, mus, sigmas):
    # Y: (batch_size, output_dim)
    # mus, sigmas: (batch_size, num_pc_per_query, output_dim)
    batch_size = mus.get_shape()[0].value
    num_pc_per_query = mus.get_shape()[1].value

    query_copies = tf.tile(query_vec, [1,int(num_pc_per_query),1])

    # Pr = 1 / ((2 * pi)^0.5 * sigma) * exp(-0.5 * ((x - mu)/sigma)^2)
    # = exp(-0.5 * ((x - mu)/sigma)^2 - log((2 * pi)^0.5 * sigma))
    # = exp(-0.5 * ((x - mu)/sigma)^2 - 0.5 * log(2 * pi) - log(sigma))
    # = exp(log_probs)

    halfLogTwoPI = 0.5 * math.log(2 * math.pi)
    queries_normalized = tf.multiply(query_copies - mus, tf.reciprocal(sigmas))
    log_probs = -0.5 * tf.square(queries_normalized) - halfLogTwoPI -\
            tf.log(sigmas)

    # Assume a diagonal covariance matrix.
    # The probability is simply the 'product' of each dimension
    # probabilities, which is 'sum' in the exponential space.
    # http://cs229.stanford.edu/section/gaussians.pdf

    # equivalent to distances
    log_probs = tf.reduce_sum(log_probs, 2)
    # log_probs: (batch_size, num_pc_per_query)

    # losses = -tf.reduce_logsumexp(log_probs, 1)
    # # losses: (N x 1)

    probs = tf.exp(-log_probs)
    # probs: (batch_size, num_pc_per_query)

    return log_probs, probs

def compute_mahalanobis(query_vec, mus, sigmas):
    batch_size = mus.get_shape()[0].value
    num_pc_per_query = mus.get_shape()[1].value

    query_copies = tf.tile(query_vec, [1,int(num_pc_per_query),1])
    queries_normalized = tf.square(tf.multiply(query_copies - mus, sigmas))
    distances = tf.reduce_sum(queries_normalized, 2)

    return distances

def compute_euclidean(query_vec, mus):
    batch_size = mus.get_shape()[0].value
    num_pc_per_query = mus.get_shape()[1].value

    query_copies = tf.tile(query_vec, [1,int(num_pc_per_query),1])
    queries_normalized = tf.square(query_copies - mus)
    distances = tf.reduce_sum(queries_normalized, 2)

    print("Euclidean")

    return distances


def distance_loss(q_vec, candidate_mus, candidate_sigmas, actual_distances, sigma=1.0):
    num_candidates = candidate_mus.get_shape()[1].value
    #(batch_size, num_candidates)
    embedding_distances = compute_mahalanobis(q_vec, candidate_mus, candidate_sigmas)

    qij = tf.nn.softmax(-embedding_distances)

    # qij_normalized_tiled = tf.tile( tf.expand_dims(tf.reduce_sum(tf.exp(-embedding_distances), axis=-1), axis=-1), [1, int(num_candidates)])
    # qij = tf.exp(-embedding_distances) / qij_normalized_tiled 

    actual_distances = actual_distances*1000 / 2.0*sigma   

    pij = tf.nn.softmax(-actual_distances)

    # pij_normalized_tiled = tf.tile( tf.expand_dims(tf.reduce_sum(tf.exp(-actual_distances / 2*sigma), axis=-1), axis=-1), [1, int(num_candidates)])
    # pij = tf.exp(-actual_distances / 2*sigma) / pij_normalized_tiled

    loss = tf.reduce_mean(tf.reduce_sum( tf.abs(qij - pij), axis = 1 ))

    return loss

def distance_regression_loss(q_vec, candidate_mus, candidate_sigmas, actual_distances, obj_sigmas, use_mahalanobis = True, use_l2 = False):
    num_candidates = candidate_mus.get_shape()[1].value
    #(batch_size, num_candidates)

    if (use_mahalanobis):
        embedding_distances = compute_mahalanobis(q_vec, candidate_mus, candidate_sigmas)
    else:
        embedding_distances = compute_euclidean(q_vec, candidate_mus)

    qij = tf.nn.softmax(-embedding_distances)

    obj_sigmas_tiled = tf.tile( tf.expand_dims(obj_sigmas, axis=-1), [1, int(num_candidates)])

    actual_distances = tf.math.divide(actual_distances, obj_sigmas_tiled)

    pij = tf.nn.softmax(-actual_distances)

    if not use_l2:
        loss = tf.reduce_mean(tf.reduce_sum( tf.abs(qij - pij), axis = 1 ))
    else:
        loss = tf.reduce_mean(tf.reduce_sum( tf.square(qij - pij), axis = 1 ))

    return loss

def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)


def distance_KL_loss(q_vec, candidate_mus, candidate_sigmas, actual_distances, obj_sigmas):
    num_candidates = candidate_mus.get_shape()[1].value
    #(batch_size, num_candidates)
    embedding_distances = compute_mahalanobis(q_vec, candidate_mus, candidate_sigmas)

    qij = tf.nn.softmax(-embedding_distances)

    obj_sigmas_tiled = tf.tile( tf.expand_dims(obj_sigmas, axis=-1), [1, int(num_candidates)])

    actual_distances = tf.math.divide(actual_distances, obj_sigmas_tiled)

    pij = tf.nn.softmax(-actual_distances)

    loss = tf.reduce_mean(kl(pij, qij))

    return loss


def embedding_loss(q_vec, pos_mus, pos_sigmas, neg_mus, neg_sigmas, margin, use_mahalanobis = True):
    # pos_distances, _ = compute_gaussian_probs(q_vec, pos_mus, pos_sigmas)
    # neg_distances, _ = compute_gaussian_probs(q_vec, neg_mus, neg_sigmas)

    if (use_mahalanobis):
        pos_distances = compute_mahalanobis(q_vec, pos_mus, pos_sigmas)
        neg_distances = compute_mahalanobis(q_vec, neg_mus, neg_sigmas)
    else:
        pos_distances = compute_euclidean(q_vec, pos_mus)
        neg_distances = compute_euclidean(q_vec, neg_mus)


    batch_num = q_vec.get_shape()[0].value
    num_neg = neg_mus.get_shape()[1].value

    best_pos = tf.reduce_max(pos_distances, 1)
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch_num), int(num_neg)],margin)

    loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m,tf.subtract(best_pos,neg_distances)), tf.zeros([int(batch_num), int(num_neg)])),1))

    return loss

def softmargin_loss(q_vec, pos_mus, pos_sigmas, neg_mus, neg_sigmas):

    pos_distances = compute_mahalanobis(q_vec, pos_mus, pos_sigmas)
    neg_distances = compute_mahalanobis(q_vec, neg_mus, neg_sigmas)

    batch_num = q_vec.get_shape()[0].value
    num_neg = neg_mus.get_shape()[1].value

    best_pos = tf.reduce_max(pos_distances, 1)
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])

    soft_loss=tf.reduce_mean(tf.reduce_sum(tf.log(tf.exp(tf.subtract(best_pos, neg_distances))+1.0), 1))
    return soft_loss

def lazy_embedding_loss(q_vec, pos_mus, pos_sigmas, neg_mus, neg_sigmas, margin):
    # pos_distances, _ = compute_gaussian_probs(q_vec, pos_mus, pos_sigmas)
    # neg_distances, _ = compute_gaussian_probs(q_vec, neg_mus, neg_sigmas)

    pos_distances = compute_mahalanobis(q_vec, pos_mus, pos_sigmas)
    neg_distances = compute_mahalanobis(q_vec, neg_mus, neg_sigmas)


    batch_num = q_vec.get_shape()[0].value
    num_neg = neg_mus.get_shape()[1].value

    best_pos = tf.reduce_max(pos_distances, 1)
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch_num), int(num_neg)],margin)

    loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m,tf.subtract(best_pos,neg_distances)), tf.zeros([int(batch_num), int(num_neg)])),1))

    return loss


def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        #batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1,int(num_pos),1]) #shape num_pos x output_dim
        # best_pos=tf.reduce_min(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos

def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
     # ''', end_points, reg_weight=0.001):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss

def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
     # ''', end_points, reg_weight=0.001):
    best_pos=best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg),1])
    best_pos=tf.tile(tf.reshape(best_pos,(-1,1)),[1, int(num_neg)])
    m=tf.fill([int(batch), int(num_neg)],margin)
    triplet_loss=tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m,tf.subtract(best_pos,tf.reduce_sum(tf.squared_difference(neg_vecs,query_copies),2))), tf.zeros([int(batch), int(num_neg)])),1))
    return triplet_loss

def chamfer_loss(pred, label):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    loss = tf.reduce_mean(dists_forward+dists_backward)
    return loss*100


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,1024,3)), outputs[1])
