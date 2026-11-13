# Winning solution: technical deep dive

## Problem framing

The challenge fixed the model family, initialization, and input size: YOLOv8n,
trained from scratch at 640 px. Ensembles, TTA, pretrained weights, and
pseudo-labeling were prohibited. The optimization surface was therefore the
dataset itself.

The four detection classes were `truck`, `car`, `van`, and `bus`, evaluated with
mAP@0.5 on a hidden test set.

## 1. Data preprocessing and quality assurance

The strongest failure mode was incomplete ground truth: visible vehicles with no
bounding box. I used the model as an error-discovery tool, not as an automatic
labeler:

1. Register the YOLO train and validation splits as versioned 3LC Tables.
2. Train a YOLOv8n baseline from random initialization.
3. In the 3LC Dashboard, filter predictions by `confidence > 0.5` and `IoU = 0`.
4. Manually inspect every candidate and promote only valid detections to ground
   truth.
5. Save a new immutable Table revision, retrain, and repeat.

This human-in-the-loop loop produced 61 table revisions and 21 training runs.
The important engineering choice was precision over automation: confidence was
used to prioritize review, never to blindly create labels.

## 2. Training strategy

The final run used YOLOv8n from scratch for 250 epochs:

| Setting | Value | Rationale |
|---|---:|---|
| Image size | 640 | Competition constraint |
| Batch size | 64 | Maximized T4 utilization |
| Optimizer | SGD | Stable scratch training |
| Initial LR | 0.01 | Combined with cosine decay |
| Momentum | 0.937 | YOLO training default retained |
| Weight decay | 0.0005 | Regularization |
| Mosaic / MixUp | 1.0 / 0.1 | Improve scale and context diversity |
| Close Mosaic | Last 20 epochs | Stabilize late-stage localization |
| Seed | 42 | Reproducible data construction and training |

For the final fit, train and validation images were consolidated, with the
validation subset repeated five times. This was a deliberate competition-stage
decision made after model selection; it should not be copied into a normal
development workflow where an untouched validation set is required.

## 3. Postprocessing and submission

Predictions were converted to the required normalized
`class confidence x_center y_center width height` representation. Class-wise NMS
used an IoU threshold of 0.5 and up to 300 detections per image.

A very low confidence cutoff (`1e-5`) preserved candidate detections for the
metric calculation and favored recall. Gamma correction was evaluated as an
image-level preprocessing ablation, but the winning submission used gamma 1.0,
meaning no correction.

## 4. Result

| Leaderboard | Score |
|---|---:|
| Public | **0.95099** |
| Private | **0.94052** |
| Final rank | **1st / 97 teams** |

The main lesson was data-centric: under a fixed model budget, systematic label
error discovery and versioned human review produced more value than architecture
tuning.
