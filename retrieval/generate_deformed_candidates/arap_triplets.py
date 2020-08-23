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

import time
from pdb import set_trace

np.random.seed(0)
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


##Sav results to file
LOG_FOUT = open('summary_arap_triplet_'+DATA_SPLIT +"_"+OBJ_CAT+'_cost.txt', 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

##For tables
if (OBJ_CAT == "table"):
	POS_THRESHOLD = 35e-5
	NEG_THRESHOLD = 75e-5
	MEDIUM_THRESHOLD = 40e-5

###For chairs
elif (OBJ_CAT == "chair"):
	POS_THRESHOLD = 30e-5
	NEG_THRESHOLD = 60e-5
	MEDIUM_THRESHOLD = 35e-5

###For sofa
elif (OBJ_CAT == "sofa"):
	POS_THRESHOLD = 20e-5
	NEG_THRESHOLD = 40e-5
	MEDIUM_THRESHOLD = 25e-5

###For car
elif (OBJ_CAT == "car"):
	POS_THRESHOLD = 12e-5
	NEG_THRESHOLD = 20e-5
	MEDIUM_THRESHOLD = 16e-5

###For airplane
elif (OBJ_CAT == "airplane"):
	POS_THRESHOLD = 5e-5
	NEG_THRESHOLD = 20e-5
	MEDIUM_THRESHOLD = 12e-5

log_string("POS_THRESHOLD: "+str(POS_THRESHOLD))
log_string("NEG_THRESHOLD: "+str(NEG_THRESHOLD))
log_string("MEDIUM_THRESHOLD: "+str(MEDIUM_THRESHOLD))


####Get candidates
pos_pickle_in = open('../../candidate_generation/candidates_'+DATA_SPLIT+'_'+OBJ_CAT+'_retrieval.pickle',"rb")
pos_candidate_idxs = pickle.load(pos_pickle_in)
pos_pickle_in.close()

neg_pickle_in = open('../../candidate_generation/candidates_'+DATA_SPLIT+'_'+OBJ_CAT+'_negatives.pickle',"rb")
neg_candidate_idxs = pickle.load(neg_pickle_in)
neg_pickle_in.close()
	        

####Get pre-computed deformed chamfer distance
FOL = "../chamfer_distance_deformed_candidates/"
pickle_in = open(os.path.join(FOL, "positive_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+".pickle"))
positive_chamfer_costs = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join(FOL, "negative_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+".pickle"))
negative_chamfer_costs = pickle.load(pickle_in)
pickle_in.close()

positives_idx = []
negatives_idx =[]
num_models_no_positives = 0
num_models_no_negatives = 0

num_pos = []
num_neg = []

for i in range(len(model_names)):
	positives = []
	negatives = []

	positive_candidates = pos_candidate_idxs[i]
	positive_costs = positive_chamfer_costs[i]
	#True positives
	idx_selected = np.argwhere((positive_costs>0) & (positive_costs<=POS_THRESHOLD)).flatten()
	[positives.append(positive_candidates[idx_selected[x]]) for x in range(idx_selected.shape[0])]
	#False positives
	idx_selected = np.argwhere((positive_costs>0) & (positive_costs>NEG_THRESHOLD)).flatten()
	[negatives.append(positive_candidates[idx_selected[x]]) for x in range(idx_selected.shape[0])]


	negative_candidates = neg_candidate_idxs[i]
	negative_costs = negative_chamfer_costs[i]
	#True negatives
	idx_selected = np.argwhere((negative_costs>0) & (negative_costs>NEG_THRESHOLD)).flatten()
	[negatives.append(negative_candidates[idx_selected[x]]) for x in range(idx_selected.shape[0])]
	#False negatives
	idx_selected = np.argwhere((negative_costs>0) & (negative_costs<=POS_THRESHOLD)).flatten()
	[positives.append(negative_candidates[idx_selected[x]]) for x in range(idx_selected.shape[0])]

	####If there is no positives, get the 5 closest deformed models
	if (len(positives)==0):
		num_to_select = 5

		closest_models_idx = np.argpartition(positive_costs, num_to_select)[:num_to_select]
		idx_selected = np.argwhere(positive_costs[closest_models_idx]<=MEDIUM_THRESHOLD).flatten()
		idx_selected = closest_models_idx[idx_selected]
		[positives.append(positive_candidates[idx_selected[x]]) for x in range(idx_selected.shape[0])]

	if (len(positives)==0):
		print("No positive samples for model "+str(i))
		num_models_no_positives += 1
	if (len(negatives)==0):
		print("No negative samples for model "+str(i))
		num_models_no_negatives += 1

	positives_idx.append(positives)
	negatives_idx.append(negatives)

	num_pos.append(len(positives))
	num_neg.append(len(negatives))

	if (i%100==0):
		print("Time elapsed: "+str(time.time()-start_time)+" sec for "+str(i)+" samples.")

# ######### Output corresponding CD cost
# positives_CD_costs = []
# negatives_CD_costs = []
# for i in range(len(positives_idx)):
# 	positives_CD_costs.append([])
# 	negatives_CD_costs.append([])

# for i in range(len(positives_idx)):
# 	positives = positives_idx[i]
# 	negatives = negatives_idx[i]

# 	positive_costs = positive_chamfer_costs[i]
# 	# print(positives)
# 	# print("")
# 	# print(pos_candidate_idxs[i])
# 	# print(positive_costs)

# 	idx_selected = np.where(np.isin(pos_candidate_idxs[i], np.array(positives)))[0]
# 	# print(idx_selected)
# 	# exit()

# 	[positives_CD_costs[i].append(positive_costs[idx_selected[x]]) for x in range(idx_selected.shape[0])]
# 	# print("")
# 	# print(pos_costs)
# 	# exit()

# 	##Append negatives
# 	idx_selected = np.where(np.isin(pos_candidate_idxs[i], np.array(negatives)))[0]
# 	[negatives_CD_costs[i].append(negative_costs[idx_selected[x]]) for x in range(idx_selected.shape[0])]


# for i in range(len(negatives_idx)):
# 	positives = positives_idx[i]
# 	negatives = negatives_idx[i]
# 	negative_costs = negative_chamfer_costs[i]

# 	# print(negatives)
# 	# print("")
# 	# print(neg_candidate_idxs[i])
# 	# print(negative_costs)

# 	idx_selected = np.where(np.isin(neg_candidate_idxs[i], np.array(negatives)))[0]
# 	[negatives_CD_costs[i].append(negative_costs[idx_selected[x]]) for x in range(idx_selected.shape[0])]

# 	# print("")
# 	# print(negatives_CD_costs[i])
# 	# exit()

# 	##Append positives
# 	idx_selected = np.where(np.isin(neg_candidate_idxs[i], np.array(positives)))[0]
# 	[positives_CD_costs[i].append(positive_costs[idx_selected[x]]) for x in range(idx_selected.shape[0])]	
# ########################################################

num_pos = np.array(num_pos)
num_neg = np.array(num_neg)
log_string(str(len(positives_idx)))
log_string("Num models no positives: "+str(num_models_no_positives))
log_string("Num models no negatives: "+str(num_models_no_negatives))
log_string("Average number of positives: "+str(np.mean(num_pos)))
log_string("Average number of negatives: "+str(np.mean(num_neg)))

# dict_value = {"positives_cost": positives_CD_costs, "negatives_cost":negatives_CD_costs}
# filename = 'arap_tripletv4_' + DATA_SPLIT + '_'+OBJ_CAT+'_cost.pickle'

dict_value = {"positives": positives_idx, "negatives":negatives_idx}
filename = 'arap_triplet_' + DATA_SPLIT + '_'+OBJ_CAT+'.pickle'

log_string("Filename: "+filename)

with open(filename, 'w') as handle:
    pickle.dump(dict_value, handle, protocol=pickle.HIGHEST_PROTOCOL)

LOG_FOUT.close()



























