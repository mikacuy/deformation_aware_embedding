import argparse
import math
import numpy as np
import os
import sys
import time
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default = "train", help='which data split to use')
parser.add_argument('--category', default='chair', help='Which class')
parser.add_argument('--perplexity', type=float, default=5.0, help='Which class')
FLAGS = parser.parse_args()

OBJ_CAT = FLAGS.category
DATA_SPLIT = FLAGS.data_split

TRAIN_CANDIDATES_FILE = "arap_distances_"+DATA_SPLIT+"_"+OBJ_CAT+".pickle"
pickle_in = open(TRAIN_CANDIDATES_FILE,"rb")
TRAIN_DICT = pickle.load(pickle_in)

PERPLEXITY = FLAGS.perplexity
threshold = 1e-5

###Candidates
candidates_idx = TRAIN_DICT["candidates"]      

###Costs
costs = TRAIN_DICT["costs"]

print(len(candidates_idx))
print(candidates_idx[0].shape)
print(len(costs))
print(costs[0].shape)


def stable_softmax(x):
	z = x - np.max(x)
	numerator = np.exp(z)
	denominator = np.sum(numerator)
	softmax = numerator/denominator
	return softmax

def calculate_perplexity(x):
	x = np.clip(x, a_min = 1e-30, a_max = None)
	log2_x = np.log2(x)
	perplexity = -np.sum(np.multiply(x, log2_x))

	return perplexity


sigmas = []
num_error = 0
for i in range(len(costs)):
	curr_cost = -costs[i]

	if (len(curr_cost) == 0 ):
		num_error += 1
		sigmas.append(-1.0)
		continue

	min_value = 0.001

	####Find appropriate min value
	transformed_cost = stable_softmax(curr_cost/min_value)
	min_perplexity = calculate_perplexity(transformed_cost)

	while (min_perplexity>PERPLEXITY):
		min_value = min_value / 2.0
		transformed_cost = stable_softmax(curr_cost/min_value)
		min_perplexity = calculate_perplexity(transformed_cost)		

	max_value = 1.0

	curr_sigma = 1
	curr_perplexity = -1
	it = 0

	while(np.abs(curr_perplexity-PERPLEXITY) > threshold):
		curr_sigma = 0.5 * (min_value + max_value)
		transformed_cost = stable_softmax(curr_cost/curr_sigma)
		curr_perplexity = calculate_perplexity(transformed_cost)

		transformed_cost = stable_softmax(curr_cost/max_value)
		max_perplexity = calculate_perplexity(transformed_cost)

		transformed_cost = stable_softmax(curr_cost/min_value)
		min_perplexity = calculate_perplexity(transformed_cost)

		if (min_perplexity < PERPLEXITY and curr_perplexity >= PERPLEXITY):
			max_value = curr_sigma
		else:
			min_value = curr_sigma
		it += 1

		if (it>200):
			print("Error in sample "+str(i))
			print(min_perplexity)
			print(max_perplexity)
			print(curr_perplexity)
			exit()

	print("Number of iterations of sample "+str(i)+"/"+str(len(costs))+": " + str(it))
	print(curr_sigma)
	print(curr_perplexity)
	print("")
	sigmas.append(curr_sigma)

sigmas = np.array(sigmas)
print(sigmas.shape)
print("Num no cost: "+str(num_error))
filename = 'arap_distances_sigmas_' + DATA_SPLIT + '_'+OBJ_CAT+'.pickle'


dict_value = {"candidates": candidates_idx, "costs":costs, "sigmas":sigmas}

print("Filename: "+filename)

with open(filename, 'w') as handle:
    pickle.dump(dict_value, handle, protocol=pickle.HIGHEST_PROTOCOL)

