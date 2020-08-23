import argparse
import math
from datetime import datetime
import os
import sys
import json
import pickle
import time
import numpy as np

np.random.seed(0)
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='chair', help='Which class')
parser.add_argument('--data_split', default = "train", help='which data split to use')
FLAGS = parser.parse_args()
OBJ_CAT = FLAGS.category
DATA_SPLIT = FLAGS.data_split

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

####Get candidates
pos_pickle_in = open('../../candidate_generation/candidates_'+DATA_SPLIT+'_'+OBJ_CAT+'_retrieval.pickle',"rb")
pos_candidate_idxs = pickle.load(pos_pickle_in)
POSITIVE_CANDIDATES_FOL = '../../manifold_top50_cd_candidates/'
pos_pickle_in.close()

neg_pickle_in = open('../../candidate_generation/candidates_'+DATA_SPLIT+'_'+OBJ_CAT+'_negatives.pickle',"rb")
neg_candidate_idxs = pickle.load(neg_pickle_in)
neg_pickle_in.close()
NEGATIVE_CANDIDATES_FOL = '../../manifold_negatives_cd_candidates/'
	        

####Get pre-computed deformed chamfer distance
FOL = "../chamfer_distance_deformed_candidates/"
pickle_in = open(os.path.join(FOL, "positive_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+".pickle"))
positive_chamfer_costs = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(os.path.join(FOL, "negative_candidates_"+DATA_SPLIT +"_"+OBJ_CAT+".pickle"))
negative_chamfer_costs = pickle.load(pickle_in)
pickle_in.close()

candidates_idx = []
candidates_costs = []

for i in range(len(model_names)):
	candidates_i = []
	candidates_costs_i = []

	positive_candidates = pos_candidate_idxs[i]
	positive_costs = positive_chamfer_costs[i]

	idx_selected = np.argwhere((positive_costs>0)).flatten()
	[candidates_i.append(positive_candidates[idx_selected[x]]) for x in range(idx_selected.shape[0])]
	[candidates_costs_i.append(positive_costs[idx_selected[x]]) for x in range(idx_selected.shape[0])]

	negative_candidates = neg_candidate_idxs[i]
	negative_costs = negative_chamfer_costs[i]

	idx_selected = np.argwhere((negative_costs>0)).flatten()
	[candidates_i.append(negative_candidates[idx_selected[x]]) for x in range(idx_selected.shape[0])]
	[candidates_costs_i.append(negative_costs[idx_selected[x]]) for x in range(idx_selected.shape[0])]

	candidates_i = np.array(candidates_i)
	candidates_costs_i = np.array(candidates_costs_i)

	candidates_idx.append(candidates_i)
	candidates_costs.append(candidates_costs_i)


print(len(candidates_idx))
print(len(candidates_costs))

dict_value = {"candidates": candidates_idx, "costs":candidates_costs}
filename = 'arap_distances_' + DATA_SPLIT + '_'+OBJ_CAT+'.pickle'
with open(filename, 'w') as handle:
    pickle.dump(dict_value, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done")