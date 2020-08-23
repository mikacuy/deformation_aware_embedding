# Deformation-Aware Embedding

Change data paths in candidate_generation/get_candidates.py and retrieval/chamfer_distance_deformed_candidates.py

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

## Comparisons:
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
First, update `SHAPENET_BASEDIR` in `generate_deformed_candidates/chamfer_triplets.py`, then run:

## Create your own training samples
To create your own data by sampling positive and negative samples and pre-computing fitting-gaps:
Sampling of positives and negatives:
```
cd candidate_generation/
python get_candidates.py --category=chair --data_split=train
python get_candidates.py --category=chair --data_split=test
python get_candidates.py --category=chair --data_split=train --generate_negatives=1
python get_candidates.py --category=chair --data_split=test --generate_negatives=1
```

Pre-computing fitting gaps:
```
cd retrieval/generate_deformed_candidates/
python chamfer_distance_deformed_candidates.py --category=chair --data_split=train
```
