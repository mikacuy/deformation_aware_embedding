# Deformation-Aware Embedding

Change data paths in candidate_generation/get_candidates.py and retrieval/chamfer_distance_deformed_candidates.py


```
cd candidate_generation/
python get_candidates.py --category=chair --data_split=train
python get_candidates.py --category=chair --data_split=test
python get_candidates.py --category=chair --data_split=train --generate_negatives=1
python get_candidates.py --category=chair --data_split=test --generate_negatives=1
```
