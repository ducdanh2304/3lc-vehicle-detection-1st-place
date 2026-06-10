# 🚗 3LC Multi-Vehicle Detection Challenge — 2nd Place Solution

**Competition:** [3LC Multi-Vehicle Detection Challenge](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge)  
**Final Rank:** 🥈 2nd Place  
**Public Score:** 0.95099 | **Private Score:** 0.94052  
**Model:** YOLOv8n (from scratch, no pretrained weights)  
**Key insight:** Data-centric iteration — repeatedly fixing missing labels via 3LC Dashboard drove the largest mAP gains.

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
Train baseline → 3LC Dashboard (Missing Label Workflow) → Add missing labels → Retrain → Repeat (many rounds)
```

### What worked
| Technique | Impact |
|-----------|--------|
| **Missing label detection via 3LC** (repeated many rounds) | Largest single gain — many vehicles were unlabeled in the original dataset |
| **Merge train + val into training set** (val repeated 5×) | More training data for final submission |
| **250 epochs with cosine LR + patience=0** | Stable convergence without early stopping |
| **SGD with momentum=0.937** | Better than default Adam for this dataset |
| **Strong augmentation** (mosaic, mixup, HSV, scale) | Improved generalization |
| **Very low conf threshold at inference (0.00001)** | Maximized recall for mAP@0.5 |

### 3LC Dashboard evidence
The `screenshots/` folder contains evidence of the label cleaning process:
- **1.webp** — 61 table revisions (EditedBoundingBox, CreatedBoundingBox, ModifiedValues) from May 3 to May 25
- **2.webp** — 21 training runs from baseline → v11, showing the full iteration history

---

## Repository Structure

```
├── train.ipynb              # Main training notebook (Kaggle T4)
├── predict_gamma.py         # Inference script (gamma=1 for final submission)
├── config.yaml              # Paths and hyperparameter config
├── dataset.yaml             # YOLO dataset config (4 classes)
├── register_tables.py       # Register 3LC Tables (from competition starter)
├── train.py                 # 3LC-integrated training script (from competition starter)
├── predict.py               # Base inference script (from competition starter)
├── verify_setup.py          # Environment check (from competition starter)
├── sample_submission.csv    # Submission template
├── screenshots/
│   ├── 1.webp               # 3LC Dashboard — 61 table revisions
│   └── 2.webp               # 3LC Dashboard — 21 training runs
└── README.md
```

> **Cleaned labels:** [Kaggle Dataset — cleaned_data_v11](https://www.kaggle.com/datasets/cdanht2/cleaned-data-v11) (labels after all 3LC cleaning rounds).  
> **Model weights:** [Kaggle Dataset — 3lc-vehicle-detection-2nd-place-weights](https://www.kaggle.com/datasets/cdanht2/3lc-vehicle-detection-2nd-place-weights).  
> **Images** (train/val/test): Download from the [competition Data tab](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/data).

---

## Environment Setup

### Requirements
- Python 3.9–3.13
- NVIDIA GPU strongly recommended
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

# 4. Install ultralytics + other deps (for train.ipynb)
pip install ultralytics opencv-python numpy

# 5. Login to 3LC (one-time)
3lc login <your_api_key>
```

Verify GPU:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## Data Preparation

1. Download competition images from the [competition Data tab](https://www.kaggle.com/competitions/3-lc-multi-vehicle-detection-challenge/data)
2. Download cleaned labels (`cleaned_data_v11.zip`) from [Kaggle Dataset](https://www.kaggle.com/datasets/cdanht2/cleaned-data-v11) and extract

Place everything at:
```
competition_starter/
├── data/
│   ├── train/
│   │   ├── images/          ← from competition
│   │   └── labels/          ← from cleaned_data_v11 (use these, not original)
│   ├── val/
│   │   ├── images/          ← from competition
│   │   └── labels/          ← from competition (original val labels)
│   └── test/
│       └── images/          ← from competition
├── dataset.yaml
├── config.yaml
└── sample_submission.csv
```

---

## 3LC Label Cleaning Workflow

This was the **core of the solution**. The competition provides imperfect labels — many vehicles appear in images without any bounding box annotation.

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

1. Go to **Runs** → select the latest run
2. Switch to the **Val** split
3. Filter: `confidence > 0.5` AND `iou = 0`  
   → high-confidence predictions with no matching ground-truth box = likely missing label
4. Review each flagged image — if the model's predicted box is correct, add it as a ground-truth label
5. Save → 3LC automatically creates a new **table revision** (data is versioned, nothing lost)

### Step 4 — Retrain with updated labels
```bash
python train.py   # automatically picks up .latest() revision
```

### Step 5 — Repeat many rounds
Repeat steps 2–4 many times until mAP stops improving. The result of all cleaning rounds is saved as `cleaned_data_v11` (61 table revisions, 21 training runs total — see `screenshots/`).

---

## Training

### Final model — `train.ipynb` (Kaggle T4)

The final model merges train + val into one training set (val repeated 5× to compensate for the smaller val size), then trains for 250 epochs with no separate validation split:

```python
EPOCHS     = 250
BATCH      = 64
IMGSZ      = 640
VAL_REPEAT = 5      # val images added 5× into training set
SEED       = 42
```

Key training settings:
```python
model.train(
    optimizer     = 'SGD',
    lr0           = 0.01,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3,
    close_mosaic  = 20,
    patience      = 0,       # no early stopping
    mixup         = 0.1,
    mosaic        = 1.0,
    scale         = 0.6,
    cos_lr        = True,
    pretrained    = False,   # from scratch — competition rule
    val           = False,   # no separate val split (merged into train)
)
```

```
Runtime  : Kaggle T4 GPU
Time     : ~2–3 hours (250 epochs, batch=64)
Output   : runs_V11_val5x_org_250ep/v11_val5x_250ep/weights/best.pt
```

### Iteration training — `train.py` (local, 3LC-integrated)
Used during the label cleaning loop (not the final submission):
```bash
cd competition_starter/
python train.py
```
Configured via `config.yaml` → `training` section.

---

## Inference

Update `predict.weights` in `config.yaml` to point to `best.pt`, then:

```bash
python predict_gamma.py
```

Output: `v11_merge_trainvalx7_250ep.csv` — rename to `submission.csv` before uploading to Kaggle.

### Key inference settings (`config.yaml`)
```yaml
predict:
  weights: /path/to/best.pt
  conf: 0.00001   # very low → maximize recall for mAP@0.5
  iou: 0.5
  imgsz: null     # uses 640 (from training)
  device: 0
  max_det: 300
  batch: 16
```

> `GAMMA_VALUE = 1` in `predict_gamma.py` = no gamma correction. The gamma experiment did not improve results; final submission used `gamma=1`.

---

## Reproducing the Final Result

### Quick path (inference only)
1. Download `best.pt` from [Kaggle Dataset](https://www.kaggle.com/datasets/cdanht2/3lc-vehicle-detection-2nd-place-weights)
2. Set `predict.weights` in `config.yaml` to the path of `best.pt`
3. Run `python predict_gamma.py` (with `GAMMA_VALUE = 1`)
4. Upload output CSV to Kaggle

### Full reproduction (train from scratch)
1. Set up environment (see [Environment Setup](#environment-setup))
2. Download competition images to `competition_starter/data/`
3. Download cleaned labels from [Kaggle Dataset](https://www.kaggle.com/datasets/cdanht2/cleaned-data-v11), extract to `data/train/labels/`
4. Run `train.ipynb` on Kaggle T4 (250 epochs, train+val merged, val×5)
5. Download `best.pt` from the notebook output
6. Set `predict.weights` in `config.yaml` to the path of `best.pt`
7. Run `python predict_gamma.py` (with `GAMMA_VALUE = 1`)
8. Upload output CSV to Kaggle

> The cleaned labels (`cleaned_data_v11`) already incorporate all label fixing done via 3LC. To reproduce the full cleaning process from scratch, start from the original competition labels and repeat the 3LC Missing Label Workflow loop many rounds using `register_tables.py` + `train.py`.

---

## Key Hyperparameters

| Parameter | Value | Note |
|-----------|-------|-------|
| Model | YOLOv8n | Competition constraint |
| Pretrained | False | Competition constraint — random init |
| Epochs | 250 | Final run |
| Batch | 64 | Kaggle T4 |
| IMGSZ | 640 | Competition constraint |
| Optimizer | SGD | |
| lr0 | 0.01 | Cosine decay |
| lrf | 0.01 | Final LR = lr0 × lrf |
| momentum | 0.937 | |
| weight_decay | 0.0005 | |
| warmup_epochs | 3 | |
| close_mosaic | 20 | Disable mosaic last 20 epochs |
| mixup | 0.1 | |
| patience | 0 | No early stopping |
| val_repeat | 5 | Val added 5× into training set |
| conf (infer) | 0.00001 | Low threshold for max recall |
| iou (infer) | 0.5 | NMS threshold |
| seed | 42 | |

---

## Acknowledgements

- [3LC](https://3lc.ai) — platform, Dashboard, and label versioning tools
- Dataset based on [UA-DETRAC benchmark](https://detrac-db.rit.albany.edu/) (Wen et al., 2020), adapted via Roboflow Universe
- Competition organized by [RishikeshJadhav3LC](https://www.kaggle.com/rishikeshjadhav3lc)
