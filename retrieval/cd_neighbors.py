import os
import sys
import numpy as np
import random
import json
import argparse
import pickle

with open('../shapenetcore_v2_split.json') as json_file:
    data = json.load(json_file)

parser = argparse.ArgumentParser()
parser.add_argument('--category', default='chair', help='Which class')
parser.add_argument('--num_neighbors', type=int, default=3, help='Number of neighbors to retrieve')
parser.add_argument('--dump_dir', default='dump_chair_ranked_cd/', help='dump folder path [dump]')

FLAGS = parser.parse_args()
OBJ_CAT = FLAGS.category
NUM_NEIGHBORS = FLAGS.num_neighbors

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

obj_list = [OBJ_CAT]

data_split = "test"
data = data[data_split]
num_categories = len(list(data.keys()))

pickle_in = open('../candidate_generation/candidates_'+data_split+'_'+OBJ_CAT+'_retrieval.pickle',"rb")
candidate_idxs = pickle.load(pickle_in)

cat_idx = -1
for i in range(num_categories):
	if (data[str(i)]["category"] == OBJ_CAT):
		cat_idx = str(i) 
		break

synsetid = data[str(cat_idx)]["synsetid"]
model_names = data[str(cat_idx)]["model_names"]
num_samples = data[str(cat_idx)]["num_samples"]

print("Synsetid: "+synsetid)
print("Num samples: "+str(num_samples))

neighbor_list = []
for i in range(len(model_names)):
	i_nbr_idx = candidate_idxs[i][:NUM_NEIGHBORS]
	neighbor_list.append(i_nbr_idx)

pickle_out = open(os.path.join(DUMP_DIR, "neighbors.pickle"),"wb")
pickle.dump(neighbor_list, pickle_out)
pickle_out.close()