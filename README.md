# 🚗 3LC Multi-Vehicle Detection Challenge — 2nd Place Solution

**Competition:** [3LC Multi-Vehicle Detection Challenge](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge)  
**Final Rank:** 🥈 2nd Place  
**Model:** YOLOv8n (from scratch, no pretrained weights)  
**Key insight:** Data-centric iteration — fixing missing labels via 3LC Dashboard drove the largest mAP gains.

---

## Table of Contents
1. [Approach Summary](#approach-summary)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [3LC Label Cleaning Workflow](#3lc-label-cleaning-workflow)
6. [Training](#training)
7. [Inference](#inference)
8. [Reproducing the Final Result](#reproducing-the-final-result)
9. [Key Hyperparameters](#key-hyperparameters)

---

## Approach Summary

The architecture is fixed (YOLOv8n, 640 px, from scratch) per competition rules, so all gains came from **data quality**.

### Iteration loop
```
Train baseline → 3LC Dashboard (Missing Label Workflow) → Add missing labels → Retrain → Repeat
```

### What worked
| Technique | Impact |
|-----------|--------|
| **Missing label detection via 3LC** | Largest single gain — many vehicles were unlabeled |
| **Merge train + val into training set** (val repeated 5×) | More training data for final submission |
| **250 epochs with cosine LR + patience=0** | Stable convergence |
| **SGD with momentum=0.937** | Better than default Adam for this dataset |
| **Strong augmentation** (mosaic, mixup, HSV, scale) | Improved generalization |
| **Low conf threshold at inference (0.00001)** | Maximized recall for mAP@0.5 |

---

## Repository Structure

```
├── train.ipynb              # Main training notebook (Kaggle)
├── predict_gamma.py         # Inference script with optional gamma preprocessing
├── config.yaml              # Paths and hyperparameter config
├── dataset.yaml             # YOLO dataset config (classes)
├── register_tables.py       # Register 3LC Tables (from competition starter)
├── train.py                 # 3LC-integrated training script (from competition starter)
├── predict.py               # Base inference script (from competition starter)
├── verify_setup.py          # Environment check (from competition starter)
├── sample_submission.csv    # Submission template
└── README.md
```

> **Note:** `data/` folder (train/val/test images + labels) is not included — download from the [competition Data tab](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/data).

---

## Environment Setup

### Requirements
- Python 3.9–3.13
- NVIDIA GPU strongly recommended (tested on RTX 3050 Ti, ~25 min/run)
- Chrome or Edge (for 3LC Dashboard)

### Install

```bash
# 1. Create virtual environment
python -m venv 3lc-env
# Windows:
3lc-env\Scripts\activate
# Linux/macOS:
source 3lc-env/bin/activate

# 2. Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install 3LC stack
pip install --index-url https://pypi.3lc.ai/public/repositories/releases-public \
            --extra-index-url https://pypi.org/simple \
            3lc-ultralytics 3lc==2.23.0 umap-learn PyYAML tqdm

# 4. Install ultralytics (for train.ipynb which uses it directly)
pip install ultralytics
pip install opencv-python numpy

# 5. Login to 3LC (one-time)
3lc login <your_api_key>
```

Verify GPU:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## Data Preparation

Download the competition starter kit from Kaggle and place it at:
```
competition_starter/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/      ← original labels (will be improved via 3LC)
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       └── images/
├── dataset.yaml
├── config.yaml
└── sample_submission.csv
```

---

## 3LC Label Cleaning Workflow

This was the **core of the solution**. The competition provides imperfect labels — many vehicles appear in images without bounding boxes.

### Step 1 — Register tables and run baseline training
```bash
cd competition_starter/
python register_tables.py
python train.py
```

### Step 2 — Start 3LC service and open Dashboard
```bash
3lc service   # keep this terminal open
# Open https://dashboard.3lc.ai in Chrome/Edge
```

### Step 3 — Find missing labels (Missing Label Workflow)
In the Dashboard, open the latest Run and apply the **Missing Label Workflow**:

1. Go to **Runs** → select your latest run
2. Switch to the **Val** split
3. Filter: `confidence > 0.5` AND `iou = 0` (high-confidence predictions with no matching ground-truth box → likely missing label)
4. Review each flagged image — if the predicted box is correct, add it as a ground-truth label
5. Save → 3LC creates a new **table revision** automatically

### Step 4 — Retrain with updated labels
```bash
python train.py   # automatically picks up .latest() revision
```

### Step 5 — Repeat
Iterate steps 2–4 until mAP stops improving. Each revision is versioned — nothing is lost.

> **Tip:** After several rounds of cleaning, the val labels become significantly more complete. The final cleaned dataset is stored as `ua_detrac_v1_cleaned` (see `config.yaml`).

---

## Training

### On Kaggle (final submission — `train.ipynb`)

The final model merges train + val into one training set (val repeated 5× to balance the split ratio), then trains for 250 epochs with no separate validation split:

```python
EPOCHS     = 250
BATCH      = 64
IMGSZ      = 640
VAL_REPEAT = 5      # val images added 5 times to training set
SEED       = 42
```

Key settings:
```python
model.train(
    optimizer     = 'SGD',
    lr0           = 0.01,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3,
    close_mosaic  = 20,
    patience      = 0,       # no early stopping for final run
    mixup         = 0.1,
    mosaic        = 1.0,
    scale         = 0.6,
    cos_lr        = True,
    pretrained    = False,   # from scratch per competition rules
    val           = False,   # merged train+val, no separate val split
)
```

Run on Kaggle (2× GPU if available, auto-detected):
```
Notebook: train.ipynb
Runtime: Kaggle GPU (P100 or T4×2)
Training time: ~2–3 hours (250 epochs, batch=64)
Output: runs_V11_val5x_org_250ep/v11_val5x_250ep/weights/best.pt
```

### Local training (3LC-integrated, for iteration)
```bash
cd competition_starter/
python train.py
```
Configured via `config.yaml` → `training` section.

---

## Inference

Update `config.yaml` → `predict.weights` to point to your `best.pt`, then:

```bash
python predict_gamma.py
```

This produces `v11_merge_trainvalx7_250ep.csv` (or whatever `OUTPUT_CSV` is set to in the script).

### Key inference settings (`config.yaml`)
```yaml
predict:
  conf: 0.00001   # very low threshold → maximize recall for mAP@0.5
  iou: 0.5
  imgsz: null     # uses training imgsz (640)
  device: 0
  max_det: 300
  batch: 16
```

> **Note on gamma:** `GAMMA_VALUE = 1` in `predict_gamma.py` means no gamma correction is applied (identity transform). The script was used to experiment with gamma values; the final submission used `gamma=1`.

Upload the output CSV to Kaggle:
```
Kaggle → Submit Predictions → upload submission CSV
```

---

## Reproducing the Final Result

### Quick path (inference only)
1. Download `best.pt` from the release (or train your own — see below)
2. Set `predict.weights` in `config.yaml` to the path of `best.pt`
3. Run `python predict_gamma.py` (with `GAMMA_VALUE = 1`)
4. Upload output CSV to Kaggle

### Full reproduction (train from scratch)
1. Set up environment (see [Environment Setup](#environment-setup))
2. Download competition data to `competition_starter/data/`
3. Register 3LC tables: `python register_tables.py`
4. Run baseline training: `python train.py`
5. Open 3LC Dashboard → apply Missing Label Workflow → add missing boxes → save revisions
6. Retrain: `python train.py` (repeat label-clean + retrain 3–5 times)
7. Copy cleaned labels to Kaggle dataset input
8. Run `train.ipynb` on Kaggle (250 epochs, train+val merged, val×5)
9. Run `python predict_gamma.py` with the output weights
10. Submit CSV

> Full reproduction takes ~3–5 hours of training + manual label review time in the Dashboard.

---

## Key Hyperparameters

| Parameter | Value | Note |
|-----------|-------|-------|
| Model | YOLOv8n | Competition constraint |
| Pretrained | False | Competition constraint |
| Epochs | 250 | Final run |
| Batch | 64 | Kaggle P100/T4 |
| IMGSZ | 640 | Competition constraint |
| Optimizer | SGD | Better than Adam here |
| lr0 | 0.01 | Cosine decay |
| lrf | 0.01 | Final LR = lr0 × lrf |
| momentum | 0.937 | YOLOv8 default |
| weight_decay | 0.0005 | |
| warmup_epochs | 3 | |
| close_mosaic | 20 | Disable mosaic last 20 ep |
| mixup | 0.1 | |
| patience | 0 | No early stopping |
| val_repeat | 5 | Val added 5× to train |
| conf (infer) | 0.00001 | Low for max recall |
| iou (infer) | 0.5 | NMS threshold |
| seed | 42 | |

---

## Acknowledgements

- [3LC](https://3lc.ai) — platform, Dashboard, and label versioning tools
- Dataset based on [UA-DETRAC benchmark](https://detrac-db.rit.albany.edu/) (Wen et al., 2020), adapted via Roboflow Universe
- Competition organized by [RishikeshJadhav3LC](https://www.kaggle.com/rishikeshjadhav3lc)
