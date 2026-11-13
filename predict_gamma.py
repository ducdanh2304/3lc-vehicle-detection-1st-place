#!/usr/bin/env python3
"""Run inference with optional gamma correction.

The winning submission used gamma=1.0 (no correction). Change GAMMA_VALUE only
when reproducing the ablation experiment.
"""
from __future__ import annotations
import csv, gc, os, shutil, sys
from pathlib import Path
import numpy as np
import cv2
import torch
from tlc_ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

GAMMA_VALUE = 1.0  # Winning configuration; try 1.2, 1.6, ... for ablations.
OUTPUT_CSV = "submission.csv"

WORK_DIR = Path(__file__).resolve().parent

# ── copy nguyên các hàm helper từ predict.py gốc ─────────────────

def _apply_ultralytics_83_compat() -> None:
    from tlc_ultralytics.detect import validator as det_val
    if getattr(det_val.TLCDetectionValidator, "_ua_detrac_compat_patched", False):
        return
    original = det_val.TLCDetectionValidator._process_detection_predictions
    def _pred_dict_to_tensor(predictions):
        if isinstance(predictions, torch.Tensor):
            return predictions
        bboxes = predictions["bboxes"]
        conf   = predictions["conf"]
        cls    = predictions["cls"]
        if bboxes.shape[0] == 0:
            return bboxes.new_empty((0, 6))
        return torch.cat([bboxes, conf.unsqueeze(1), cls.unsqueeze(1).float()], dim=1)
    def patched(self, preds, batch):
        from tlc_ultralytics.detect.utils import construct_bbox_struct
        from ultralytics.utils import metrics, ops
        predicted_boxes = []
        for i, predictions in enumerate(preds):
            predictions = _pred_dict_to_tensor(predictions)
            ori_shape    = batch["ori_shape"][i]
            resized_shape = batch["resized_shape"][i]
            ratio_pad    = batch["ratio_pad"][i]
            height, width = ori_shape
            if len(predictions) == 0:
                predicted_boxes.append(construct_bbox_struct([], image_width=width, image_height=height))
                continue
            predictions = predictions.clone()
            predictions = predictions[predictions[:, 4] > self._settings.conf_thres]
            predictions = predictions[predictions[:, 4].argsort(descending=True)[:self._settings.max_det]]
            pred_box    = predictions[:, :4].clone()
            pred_scaled = ops.scale_boxes(resized_shape, pred_box, ori_shape, ratio_pad)
            pbatch  = self._prepare_batch(i, batch)
            gt_bbox = pbatch.get("bbox", pbatch.get("bboxes"))
            if gt_bbox is not None and gt_bbox.shape[0]:
                ious     = metrics.box_iou(gt_bbox, pred_scaled)
                box_ious = ious.max(dim=0)[0].cpu().tolist()
            else:
                box_ious = [0.0] * pred_scaled.shape[0]
            pred_xywh = ops.xyxy2xywhn(pred_scaled, w=width, h=height)
            conf_list = predictions[:, 4].cpu().tolist()
            pred_cls  = predictions[:, 5].cpu().tolist()
            annotations = [
                {"score": conf_list[pi],
                 "category_id": self.data["range_to_3lc_class"][int(pred_cls[pi])],
                 "bbox": pred_xywh[pi, :].cpu().tolist(),
                 "iou": box_ious[pi]}
                for pi in range(len(predictions))
            ]
            predicted_boxes.append(construct_bbox_struct(annotations, image_width=width, image_height=height))
        return predicted_boxes
    det_val.TLCDetectionValidator._process_detection_predictions = patched
    det_val.TLCDetectionValidator._ua_detrac_compat_patched = True

def _load_config() -> dict:
    import yaml
    p = WORK_DIR / "config.yaml"
    if not p.is_file():
        print(f"ERROR: Missing {p}", file=sys.stderr); sys.exit(1)
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (WORK_DIR / p).resolve()

def _resolve_weights(cfg: dict) -> Path:
    paths_cfg   = cfg.get("paths", {})
    training    = cfg.get("training", {})
    predict_cfg = cfg.get("predict", {})
    runs_root   = str(paths_cfg.get("runs_detect_root", "runs/detect"))
    run_name    = str(training.get("run_name", "yolov8n_baseline"))
    explicit    = predict_cfg.get("weights")
    if explicit:
        p = _resolve(str(explicit))
        if p.is_file(): return p
    primary = WORK_DIR / runs_root / run_name / "weights" / "best.pt"
    if primary.is_file(): return primary.resolve()
    candidates = sorted(
        (WORK_DIR / runs_root).glob("*/weights/best.pt"),
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    if candidates: return candidates[0].resolve()
    raise FileNotFoundError("No weights found.")

def _result_to_prediction_string(result) -> str:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "no box"
    xywhn = boxes.xywhn.cpu().numpy()
    cls   = boxes.cls.cpu().numpy().astype(int)
    conf  = boxes.conf.cpu().numpy()
    order = np.argsort(-conf)
    parts: list[str] = []
    for i in order:
        c = int(cls[i]); cf = float(conf[i])
        x, y, w, h = (float(v) for v in xywhn[i])
        x = min(1.0, max(0.0, x)); y = min(1.0, max(0.0, y))
        w = min(1.0, max(0.0, w)); h = min(1.0, max(0.0, h))
        parts.extend([str(c), f"{cf:.6f}", f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"])
    return " ".join(parts)

def _find_image(test_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = test_dir / f"{stem}{ext}"
        if p.is_file(): return p
    return None

# ── gamma preprocessing ───────────────────────────────────────────

def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = (np.arange(256) / 255.0) ** (1.0 / gamma) * 255
    return cv2.LUT(img, lut.astype(np.uint8))

def preprocess_to_tmp(test_dir: Path, stems: list[str], gamma: float) -> Path:
    """Apply gamma lên toàn bộ test images, lưu vào tmp folder."""
    tmp_dir = WORK_DIR / f"_tmp_gamma_{gamma}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"  Applying gamma={gamma} to {len(stems)} images → {tmp_dir.name}/")
    missing = 0
    for stem in tqdm(stems, desc="Preprocessing", unit="img"):
        src = _find_image(test_dir, stem)
        if src is None:
            missing += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            missing += 1
            continue
        out = apply_gamma(img, gamma)
        cv2.imwrite(str(tmp_dir / src.name), out)

    if missing:
        print(f"  Warning: {missing} images not found/readable")
    return tmp_dir

# ── inference (memory pipeline) ───────────────────────────────────

def _pipeline_memory(
    model, test_dir: Path, stems: list[str],
    *, imgsz, conf, iou, device, max_det, batch,
) -> dict[str, str]:
    paths = [p for stem in stems if (p := _find_image(test_dir, stem)) is not None]
    pred_by_stem: dict[str, str] = {}
    bs = max(1, int(batch))

    for start in tqdm(range(0, len(paths), bs), desc="Inferring", unit="batch"):
        chunk   = paths[start:start + bs]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=int(imgsz), conf=float(conf), iou=float(iou),
            device=device, max_det=int(max_det),
            batch=min(bs, len(chunk)), verbose=False,
        )
        for res, p in zip(results, chunk):
            pred_by_stem[p.stem] = _result_to_prediction_string(res)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pred_by_stem

# ── main ──────────────────────────────────────────────────────────

def main() -> int:
    os.chdir(WORK_DIR)
    _apply_ultralytics_83_compat()

    cfg         = _load_config()
    paths_cfg   = cfg.get("paths", {})
    training    = cfg.get("training", {})
    predict_cfg = cfg.get("predict", {})

    sample_path = _resolve(str(paths_cfg.get("sample_submission", "sample_submission.csv")))
    out_path    = WORK_DIR / OUTPUT_CSV
    test_dir    = _resolve(str(paths_cfg.get("test_images", "data/test/images")))

    conf    = float(predict_cfg.get("conf", 0.25))
    iou     = float(predict_cfg.get("iou", 0.7))
    imgsz   = int(predict_cfg.get("imgsz") or training.get("image_size", 640))
    device  = predict_cfg.get("device", training.get("device", 0))
    max_det = int(predict_cfg.get("max_det", 300))
    batch   = int(predict_cfg.get("batch", 1))

    print("=" * 70)
    print(f"PREDICT WITH GAMMA={GAMMA_VALUE}  →  {OUTPUT_CSV}")
    print("=" * 70)

    if not sample_path.is_file():
        print(f"ERROR: {sample_path} not found", file=sys.stderr); return 1
    if not test_dir.is_dir():
        print(f"ERROR: {test_dir} not found",    file=sys.stderr); return 1

    weights = _resolve_weights(cfg)

    rows_in: list[dict] = []
    with sample_path.open(newline="", encoding="utf-8") as f:
        rows_in.extend(csv.DictReader(f))
    stems = [str(r["image_id"]).strip() for r in rows_in]

    print(f"\n  Rows      : {len(rows_in)}")
    print(f"  Test dir  : {test_dir}")
    print(f"  Weights   : {weights}")
    print(f"  conf={conf}, iou={iou}, imgsz={imgsz}, batch={batch}")

    # 1. preprocess → tmp
    tmp_dir = preprocess_to_tmp(test_dir, stems, GAMMA_VALUE)

    # 2. load model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model = YOLO(str(weights))
    print("  Model loaded.\n")

    # 3. infer trên ảnh đã preprocess
    pred_by_stem = _pipeline_memory(
        model, tmp_dir, stems,
        imgsz=imgsz, conf=conf, iou=iou,
        device=device, max_det=max_det, batch=batch,
    )

    # 4. cleanup tmp
    shutil.rmtree(tmp_dir)
    print(f"\n  Cleaned up {tmp_dir.name}")

    # 5. ghi CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_id", "prediction_string"])
        w.writeheader()
        for row, stem in zip(rows_in, stems):
            w.writerow({
                "id": row["id"],
                "image_id": stem,
                "prediction_string": pred_by_stem.get(stem, "no box"),
            })

    nonempty = sum(1 for s in stems if pred_by_stem.get(s, "no box") != "no box")
    n_boxes  = sum(
        len(pred_by_stem.get(s, "").split()) // 6
        for s in stems if pred_by_stem.get(s, "no box") != "no box"
    )

    print("\n" + "=" * 70)
    print("OK — SUBMISSION READY")
    print("=" * 70)
    print(f"  File   : {out_path}")
    print(f"  Gamma  : {GAMMA_VALUE}")
    print(f"  Images with box : {nonempty} / {len(rows_in)}")
    print(f"  Total boxes     : {n_boxes}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
