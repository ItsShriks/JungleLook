# Retrospective: v1 (R&D prototype) → v2 (this pipeline)

The first version of this project was exploratory notebooks and scripts. Auditing it before the rewrite
turned up problems that explain its results, and each one shaped a design decision in v2.

| v1 issue | effect | v2 decision |
|---|---|---|
| Grid-cell training labels were drawn **at random** in fixed proportions (`assign_labels_by_percentage`) | nothing learnable; the network could only learn class priors | labels come from explicit, inspectable geometric rules (`pseudolabel.py`) and are scored against real annotations |
| The "PointNet++" model was a per-point MLP with no neighbourhood or global context | each point classified from its own xyz alone | real PointNet (T-Nets, global feature) and PointNet++ (set abstraction / feature propagation), both unit-tested on a toy task |
| Validation used the training files | reported 93 % was training accuracy; held-out cell accuracy was 11.8 % | spatial tile split into train / val / test; test is only touched by `evaluate` |
| Zero-padding to a fixed point count added fake label-0 points | inflated accuracy, biased the loss | resampling within blocks + ignore index; sliding-window voting at inference |
| Classifier: 10 output classes for 3, 66 training samples, points resampled from Delaunay meshes of themselves | unstable, near-chance validation | dropped image-style classification in favour of point-wise segmentation + instance extraction |
| Annotations (UTM orthophoto) and LiDAR (local SLAM frame) were never aligned in code | stump labels could not be used | `junglelook register`: RANSAC + ICP, 98.8 % inliers |
| Hard-coded absolute paths, one script per step | not reproducible | one CLI, YAML configs with overrides, checkpoints that carry their config and normalisation |
