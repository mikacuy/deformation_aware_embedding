import os
import numpy as np
import random
import h5py
import json

BASE_PATH = "/orion/u/mikacuy/data/ShapeNetCore.v2.PC15k/"

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

### Train/Test split gotten from here: https://github.com/stevenygd/PointFlow
def get_train_test_split():
    data_splits = ["train", "val", "test"]
    obj_classes = ["chair", "table", "bench", "sofa", "cabinet", "bookshelf", "car", "airplane"]
    print(obj_classes)

    splitted_data = {}
    splitted_data["categories"] = obj_classes
    for data_split in data_splits:
        splitted_data[data_split] = {}

        for i in range(len(obj_classes)):
            obj = obj_classes[i]
            obj_fol = cate_to_synsetid[obj]
            model_path = os.path.join(BASE_PATH, obj_fol, data_split)
            model_files = sorted(os.listdir(model_path))

            #Append number of models
            splitted_data[data_split][i] = {}
            splitted_data[data_split][i]["category"] = obj
            splitted_data[data_split][i]["synsetid"] = obj_fol

            model_names = []
            num_samples = 0
            for model_file in model_files:
                if ("._" in model_file):
                    continue

                name = model_file.split('.')[0]
                model_names.append(name)
                num_samples +=1

            splitted_data[data_split][i]["num_samples"] = num_samples
            splitted_data[data_split][i]["model_names"] = model_names

    with open('shapenetcore_v2_split.json', 'w') as outfile:
        json.dump(splitted_data, outfile, indent=2)

get_train_test_split()







