import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import transformations

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_rotation_matrices(num_samples):
    rotation_matrices = np.zeros((num_samples,3,3), dtype=np.float32)
    for k in range(num_samples):
      random_rot_matrix= transformations.random_rotation_matrix()[:-1,:-1]
      rotation_matrices[k, ...] = random_rot_matrix

    return rotation_matrices

def rotate_point_cloud_by_matrices(batch_data, rotation_matrices):
    """ Rotate the point cloud corresponding to each rotation matrix.
        Input:
          batch_data = BxNx3 array, original batch of point clouds
          rotation_matrices = Bx3x3, corresponding rotation matrix per point cloud
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
      shape_pc = batch_data[k, ...]
      rotation_matrix = rotation_matrices[k, ...]
      rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def get_current_data(pcs, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]
    #sampled = pcs[:,:num_points,:]

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(pcs.shape[0])
        np.random.shuffle(idx)
        sampled = sampled[idx]

    return sampled

def get_current_data_triplet(pcs, dict_values, num_pos, num_neg, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]

    ###Get query
    query = sampled

    ###Positives
    positives = dict_values["positives"]
    idx_pts = np.arange(positives.shape[1])
    np.random.shuffle(idx_pts)
    positives = positives[:, idx_pts[:num_pos]]        

    ###Negatives
    negatives = dict_values["negatives"]
    idx_pts = np.arange(negatives.shape[1])
    np.random.shuffle(idx_pts)
    negatives = negatives[:, idx_pts[:num_neg]]

    ##Get point clouds
    positive_pcs = []
    negative_pcs = []
    for i in range(pcs.shape[0]):
        positive_pcs.append(sampled[positives[i],:,:])
        negative_pcs.append(sampled[negatives[i],:,:])

    positive_pcs = np.array(positive_pcs)
    negative_pcs = np.array(negative_pcs)

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(pcs.shape[0])
        np.random.shuffle(idx)
        query = query[idx]
        positive_pcs = positive_pcs[idx]
        negative_pcs = negative_pcs[idx]

    return query, positive_pcs, negative_pcs

def get_current_data_arap(pcs, dict_values, num_pos, num_neg, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]

    ###Get query
    query = sampled

    ###Positives
    positives = dict_values["positives"]      

    ###Negatives
    negatives = dict_values["negatives"]

    ##Get point clouds
    positive_pcs = []
    negative_pcs = []
    idx_to_keep = []
    for i in range(pcs.shape[0]):
        positive_idx =  positives[i]
        negative_idx = negatives[i]

        if (len(positive_idx) == 0 or len(negative_idx) == 0):
            continue

        idx_to_keep.append(i)

        #select idx
        positive_idx_selected = np.random.choice(positive_idx, num_pos, replace=True)
        negative_idx_selected = np.random.choice(negative_idx, num_neg, replace=True)

        positive_pcs.append(sampled[positive_idx_selected,:,:])
        negative_pcs.append(sampled[negative_idx_selected,:,:])

    positive_pcs = np.array(positive_pcs)
    negative_pcs = np.array(negative_pcs)
    idx_to_keep = np.array(idx_to_keep)
    query = query[idx_to_keep]

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(query.shape[0])
        np.random.shuffle(idx)
        query = query[idx]
        positive_pcs = positive_pcs[idx]
        negative_pcs = negative_pcs[idx]

        return query, positive_pcs, negative_pcs, idx_to_keep[idx]

    return query, positive_pcs, negative_pcs 


def get_current_data_arap_scan2cad(pcs, scans, dict_values, num_pos, num_neg, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]

    idx_pts = np.arange(scans.shape[1])
    np.random.shuffle(idx_pts)
    scans_sampled =  scans[:,idx_pts[:num_points],:]


    ###Get query
    query = scans_sampled

    ###Positives
    positives = dict_values["positives"]      

    ###Negatives
    negatives = dict_values["negatives"]

    ##Get point clouds
    positive_pcs = []
    negative_pcs = []
    idx_to_keep = []
    for i in range(scans.shape[0]):
        positive_idx =  positives[i]
        negative_idx = negatives[i]

        if (len(positive_idx) == 0 or len(negative_idx) == 0):
            continue

        idx_to_keep.append(i)

        #select idx
        positive_idx_selected = np.random.choice(positive_idx, num_pos, replace=True)
        negative_idx_selected = np.random.choice(negative_idx, num_neg, replace=True)

        positive_pcs.append(sampled[positive_idx_selected,:,:])
        negative_pcs.append(sampled[negative_idx_selected,:,:])

    positive_pcs = np.array(positive_pcs)
    negative_pcs = np.array(negative_pcs)
    idx_to_keep = np.array(idx_to_keep)
    query = query[idx_to_keep]

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(query.shape[0])
        np.random.shuffle(idx)
        query = query[idx]
        positive_pcs = positive_pcs[idx]
        negative_pcs = negative_pcs[idx]

        return query, positive_pcs, negative_pcs, idx_to_keep[idx]

    return query, positive_pcs, negative_pcs    

def get_current_data_arap_scan2cad_distances_with_sigmas(pcs, scans, dict_values, num_candidates, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]

    idx_pts = np.arange(scans.shape[1])
    np.random.shuffle(idx_pts)
    scans_sampled =  scans[:,idx_pts[:num_points],:]

    ###Get query
    query = scans_sampled

    ###Candidates
    candidates = dict_values["candidates"]      

    ###Costs
    costs = dict_values["costs"]

    ###Sigmas
    sigmas = dict_values["sigmas"]    

    ##Get point clouds
    candidates_pcs = []
    candidates_costs = []
    idx_to_keep = []
    for i in range(scans.shape[0]):
        candidates_idx_i =  candidates[i]
        candidates_cost_i = np.squeeze(costs[i])

        if (candidates_idx_i.shape[0] < num_candidates):
            continue

        if (candidates_idx_i.shape[0] != candidates_cost_i.shape[0]):
            print("Error in data.")
            exit()

        idx_to_keep.append(i)

        #select idx
        num_options = candidates_idx_i.shape[0]
        idx = np.arange(num_options)
        np.random.shuffle(idx)
        to_select = idx[:num_candidates]

        candidates_pcs.append(sampled[candidates_idx_i[to_select],:,:])
        candidates_costs.append(candidates_cost_i[to_select])

    candidates_pcs = np.array(candidates_pcs)
    candidates_costs = np.array(candidates_costs)
    idx_to_keep = np.array(idx_to_keep)
    query = query[idx_to_keep]
    sigmas = sigmas[idx_to_keep]

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(query.shape[0])
        np.random.shuffle(idx)
        query = query[idx]
        sigmas = sigmas[idx]
        candidates_pcs = candidates_pcs[idx]
        candidates_costs = candidates_costs[idx]

        return query, candidates_pcs, candidates_costs, sigmas, idx_to_keep[idx]

    return query, positive_pcs, negative_pcs    

def get_current_data_arap_scan2cad_distances(pcs, scans, dict_values, num_candidates, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]

    idx_pts = np.arange(scans.shape[1])
    np.random.shuffle(idx_pts)
    scans_sampled =  scans[:,idx_pts[:num_points],:]

    ###Get query
    query = scans_sampled

    ###Candidates
    candidates = dict_values["candidates"]      

    ###Costs
    costs = dict_values["costs"]
   

    ##Get point clouds
    candidates_pcs = []
    candidates_costs = []
    idx_to_keep = []
    for i in range(scans.shape[0]):
        candidates_idx_i =  candidates[i]
        candidates_cost_i = np.squeeze(costs[i])

        if (candidates_idx_i.shape[0] < num_candidates):
            continue

        if (candidates_idx_i.shape[0] != candidates_cost_i.shape[0]):
            print("Error in data.")
            exit()

        idx_to_keep.append(i)

        #select idx
        num_options = candidates_idx_i.shape[0]
        idx = np.arange(num_options)
        np.random.shuffle(idx)
        to_select = idx[:num_candidates]

        candidates_pcs.append(sampled[candidates_idx_i[to_select],:,:])
        candidates_costs.append(candidates_cost_i[to_select])

    candidates_pcs = np.array(candidates_pcs)
    candidates_costs = np.array(candidates_costs)
    idx_to_keep = np.array(idx_to_keep)
    query = query[idx_to_keep]

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(query.shape[0])
        np.random.shuffle(idx)
        query = query[idx]
        candidates_pcs = candidates_pcs[idx]
        candidates_costs = candidates_costs[idx]

        return query, candidates_pcs, candidates_costs, idx_to_keep[idx]

    return query, positive_pcs, negative_pcs    

def get_current_data_arap_distances(pcs, dict_values, num_candidates, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]

    ###Get query
    query = sampled

    ###Candidates
    candidates = dict_values["candidates"]      

    ###Costs
    costs = dict_values["costs"]

    ##Get point clouds
    candidates_pcs = []
    candidates_costs = []
    idx_to_keep = []
    for i in range(pcs.shape[0]):
        candidates_idx_i =  candidates[i]
        candidates_cost_i = np.squeeze(costs[i])

        if (candidates_idx_i.shape[0] < num_candidates):
            continue

        if (candidates_idx_i.shape[0] != candidates_cost_i.shape[0]):
            print("Error in data.")
            exit()

        idx_to_keep.append(i)

        #select idx
        num_options = candidates_idx_i.shape[0]
        idx = np.arange(num_options)
        np.random.shuffle(idx)
        to_select = idx[:num_candidates]

        candidates_pcs.append(sampled[candidates_idx_i[to_select],:,:])
        candidates_costs.append(candidates_cost_i[to_select])

    candidates_pcs = np.array(candidates_pcs)
    candidates_costs = np.array(candidates_costs)
    idx_to_keep = np.array(idx_to_keep)
    query = query[idx_to_keep]

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(query.shape[0])
        np.random.shuffle(idx)
        query = query[idx]
        candidates_pcs = candidates_pcs[idx]
        candidates_costs = candidates_costs[idx]

        return query, candidates_pcs, candidates_costs, idx_to_keep[idx]


    return query, candidates_pcs, candidates_costs 

def get_current_data_arap_distances_with_sigmas(pcs, dict_values, num_candidates, num_points, shuffle=True):
    #shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:,idx_pts[:num_points],:]

    ###Get query
    query = sampled

    ###Candidates
    candidates = dict_values["candidates"]      

    ###Costs
    costs = dict_values["costs"]

    ###Sigmas
    sigmas = dict_values["sigmas"]

    ##Get point clouds
    candidates_pcs = []
    candidates_costs = []
    idx_to_keep = []
    for i in range(pcs.shape[0]):
        candidates_idx_i =  candidates[i]
        candidates_cost_i = np.squeeze(costs[i])

        if (candidates_idx_i.shape[0] < num_candidates):
            continue

        if (candidates_idx_i.shape[0] != candidates_cost_i.shape[0]):
            print("Error in data.")
            exit()

        idx_to_keep.append(i)

        #select idx
        num_options = candidates_idx_i.shape[0]
        idx = np.arange(num_options)
        np.random.shuffle(idx)
        to_select = idx[:num_candidates]

        candidates_pcs.append(sampled[candidates_idx_i[to_select],:,:])
        candidates_costs.append(candidates_cost_i[to_select])

    candidates_pcs = np.array(candidates_pcs)
    candidates_costs = np.array(candidates_costs)
    idx_to_keep = np.array(idx_to_keep)
    query = query[idx_to_keep]
    sigmas = sigmas[idx_to_keep]

    #shuffle point clouds per epoch
    if (shuffle):
        idx = np.arange(query.shape[0])
        np.random.shuffle(idx)
        query = query[idx]
        sigmas = sigmas[idx]
        candidates_pcs = candidates_pcs[idx]
        candidates_costs = candidates_costs[idx]

        return query, candidates_pcs, candidates_costs, sigmas, idx_to_keep[idx]


    return query, candidates_pcs, candidates_costs 


def load_h5(h5_filename):
    f = h5py.File(h5_filename, "r")
    data = f['data'][:]
    return data


