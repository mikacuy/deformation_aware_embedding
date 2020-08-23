# Deformation-Aware Embedding

## Data download
Download h5 files for each object class (chair, table, sofa, car, airplane), and sampled positives and negatives for each model with pre-computed fitting gaps:
TODO

Goto `retrieval/` for training/test of our networks and the baselines.

## Our Deformation-Aware Embedding
### Ours-Reg:
1) Construct data for network training/test for Ours-Reg:
```
cd generate_deformed_candidates/
python arap_distances.py --category=chair --data_split=train
python get_object_sigmas.py --category=chair --data_split=train
```
2) Network training:
```
python train_ours_distances.py --category=chair
```
3) Retrieval:
Select the desired model and result directory with flags `--model_path` and `--dump_dir`
```
python retrieval_gaussian.py --category=chair
```

### Ours-Margin:
1) Construct data for network training/test for Ours-Margin:
```
cd generate_deformed_candidates/
python arap_triplets.py --category=chair --data_split=train
```
2) Network training:
```
python train_ours_triplet.py --category=chair
```
3) Retrieval:
Select the desired model and result directory with flags `--model_path` and `--dump_dir`
```
python retrieval_gaussian.py --category=chair
```

## Baselines:
### Ranked-CD:
1) Retrieval:
```
python cd_neighbors.py --category=chair
```

### AE:
1) Network training:
```
python train_autoencoder.py --category=chair
```
3) Retrieval:
Select the desired model and result directory with flags `--model_path` and `--dump_dir`
```
python retrieval.py --category=chair --model=pointnet_autoencoder
```

### CD-Triplet:
1) Construct triplets based on chamfer distances. First, update `SHAPENET_BASEDIR` in `generate_deformed_candidates/chamfer_triplets.py`, then run:
```
cd generate_deformed_candidates/
python chamfer_triplets.py --category=chair --data_split=train
```
2) Network training:
```
python train_triplet.py --category=chair
```
3) Retrieval:
Select the desired model and result directory with flags `--model_path` and `--dump_dir`
```
python retrieval.py --category=chair --model=pointnet_triplet
```

## Evaluation of fitting gap (post-deformation chamfer distance)
### Pre-requisites
#### Compile deformation function
In our experiments we chose to use a simplest version of deformation function found [here](https://github.com/hjwdzh/MeshODE). Please follow the installation pre-requisites found in this [repo](https://github.com/hjwdzh/MeshODE) to build.
```
cd ../meshdeform
mkdir build 
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```
#### Compile point-to-mesh distance function
In our paper, we report the point-to-mesh distance for fitting loss (an alternative approximate is to use a point cloud to point cloud distance). To use the point-to-mesh distance metric, first compile the function by running:
```
cd ../tools/evaluation
mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Scripts
Update `SHAPENET_BASEDIR` in `evaluate_fitting_deform_fast.py`, `evaluate.py`, `evaluate_point2mesh` and `evaluate_testrank.py`, and select the desired  result directory with flag`--dump_dir` for all the evaluation scripts below.

Run for fitting error post-deformation:
```
python evaluate_fitting_deform_fast.py --category=chair
```

For the ranking evaluation found in our paper, select the desired model and result directory with flags `--model_path` and `--dump_dir`, and then run:
```
python retrieval_gaussian.py --category=chair --testrank=1
python evaluate_testrank.py --category=chair
```

For fitting error without deformation, run:
```
python evaluate_point2mesh.py --category=chair --fitting_dump_dir=point2mesh_new2_nodeform/
```

## Create your own training samples
To create your own data by sampling positive and negative samples and pre-computing fitting-gaps, first change `SHAPENET_BASEDIR` in candidate_generation/get_candidates.py and `POSITIVE_CANDIDATES_FOL` and `NEGATIVE_CANDIDATES_FOL` in retrieval/chamfer_distance_deformed_candidates.py. Then run:

1) Sampling of positives and negatives
```
cd candidate_generation/
python get_candidates.py --category=chair --data_split=train
python get_candidates.py --category=chair --data_split=test
python get_candidates.py --category=chair --data_split=train --generate_negatives=1
python get_candidates.py --category=chair --data_split=test --generate_negatives=1
```

2) Pre-computing fitting gaps
```
cd retrieval/generate_deformed_candidates/
python chamfer_distance_deformed_candidates.py --category=chair --data_split=train
```
