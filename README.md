# Deformation-Aware Embedding

Change data paths in candidate_generation/get_candidates.py and retrieval/chamfer_distance_deformed_candidates.py

Download h5 files for each object class, and sampled positives and negatives for each model with pre-computed fitting gaps:
TODO

Goto `retrieval/` for network training/test.

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

Comparisons:
### AE:
```
python train_autoencoder.py --category=chair
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
