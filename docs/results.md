# Results on Field-D

**Data.** A UAV LiDAR SLAM map of a clear-cut (Garrulus), with a 68 × 80 m region of interest: 725 k points,
~130 pts/m², and no RGB. Reference annotations are 129 stump polygons digitised on the RGB orthophoto
(UTM). They are used only for evaluation.

## Registration of the annotations

| | |
|---|---|
| RANSAC centroid matches | 96 / 127 stumps (chance level ≈ 45) |
| rotation / tilt | 79.24° / 0.13° |
| vertical offset consistency | MAD 9 cm across all 127 stumps |
| ICP | **98.8 %** of annotation points within 0.3 m of the LiDAR surface, RMS 0.11 m |

## Ground model

This was checked against CloudCompare's Cloth Simulation Filter output for the same ROI. CSF labels everything
within ~0.5 m of the cloth as ground. At that level, the height above our DTM separates the CSF classes almost
perfectly: 99 % of CSF-ground points lie below 0.65 m and 99 % of CSF-off-ground points lie above 0.43 m.

## Pseudo-labels vs. the registered stump annotations

| | value |
|---|---|
| stump instances: precision / recall / F1 | 0.27 / 0.36 / **0.31** |
| stump points: IoU | 0.11 |
| not-stump points: IoU | 0.94 |

## Network (preliminary)

PointNet++ was trained on the pseudo-labels only. The run was interrupted at epoch 11 of 20.

| val (vs pseudo-labels) | mIoU | ground | stump | other |
|---|---|---|---|---|
| best epoch (9) | **0.80** | 0.99 | 0.46 | 0.95 |

Evaluation of the network against the registered annotations is pending. Run it with
`junglelook run -c configs/field_d.yaml`. `configs/field_d.yaml` with `pseudolabel.stumps_from_reference=true`
gives the supervised upper bound on the same test tiles.

## Why stumps are hard here

The limit is the data, and it can be measured:

* After removing a local plane per 0.5 m cell, the **ground layer is ~26 cm thick** (median 5–95 % spread,
  SLAM registration noise plus low vegetation).
* Annotated stumps rise only **0.2–0.4 m** (median 0.27 m) above the ground and get roughly 15–20 LiDAR
  points each.
* The feature distributions of stump points and other low off-ground points almost coincide (median HAG
  0.20 vs 0.28 m, the same scattering and planarity).

A stump is therefore about as tall as the ground noise. Denser data should separate them: the RGB photogrammetry
cloud of the same field has 143 M points, and TLS/MLS scans would also help. The pipeline takes such clouds unchanged.
