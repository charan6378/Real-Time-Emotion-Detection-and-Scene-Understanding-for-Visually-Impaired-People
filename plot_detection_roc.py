import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import auc, roc_curve
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return [x1, y1, x2, y2]


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_dataset(dataset_dir: Path):
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, dict):
            class_names = [names[i] for i in sorted(names.keys())]
        else:
            class_names = list(names)
        images_dir = dataset_dir / "test" / "images"
        labels_dir = dataset_dir / "test" / "labels"
    else:
        # Support tiny datasets like coco8 that ship as images/val + labels/val.
        class_names = [str(i) for i in range(80)]
        images_dir = dataset_dir / "images" / "val"
        labels_dir = dataset_dir / "labels" / "val"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected image/label dirs not found: {images_dir} and {labels_dir}"
        )

    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    if not image_paths:
        raise ValueError("No test images found.")

    dataset = []
    for img_path in image_paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    gt_boxes.append((cls, yolo_to_xyxy(xc, yc, w, h, img_w, img_h)))
        dataset.append((img_path, gt_boxes))

    return class_names, dataset


def collect_predictions(model, dataset, num_classes):
    records = []
    for img_path, gt in dataset:
        gt_by_class = {c: [] for c in range(num_classes)}
        for cls, box in gt:
            if 0 <= cls < num_classes:
                gt_by_class[cls].append(box)

        pred_by_class = {c: [] for c in range(num_classes)}
        result = model(str(img_path), verbose=False)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            pred_cls = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf = result.boxes.conf.cpu().numpy()
            pred_xyxy = result.boxes.xyxy.cpu().numpy()
            for cls, conf, box in zip(pred_cls, pred_conf, pred_xyxy):
                if 0 <= cls < num_classes:
                    pred_by_class[cls].append((float(conf), box.tolist()))
        records.append((gt_by_class, pred_by_class))
    return records


def build_roc_inputs(records, num_classes, iou_thr):
    y_true = {c: [] for c in range(num_classes)}
    y_score = {c: [] for c in range(num_classes)}

    for gt_by_class, pred_by_class in records:
        for c in range(num_classes):
            gt_boxes = gt_by_class[c]
            preds = sorted(pred_by_class[c], key=lambda x: x[0], reverse=True)
            matched_gt = set()

            for conf, pbox in preds:
                best_iou = 0.0
                best_idx = -1
                for i, gbox in enumerate(gt_boxes):
                    if i in matched_gt:
                        continue
                    iou_val = iou(pbox, gbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = i
                is_tp = best_iou >= iou_thr and best_idx >= 0
                if is_tp:
                    matched_gt.add(best_idx)
                y_true[c].append(1 if is_tp else 0)
                y_score[c].append(conf)

            # Missed objects are positives at zero confidence (false negatives).
            missed = len(gt_boxes) - len(matched_gt)
            for _ in range(max(0, missed)):
                y_true[c].append(1)
                y_score[c].append(0.0)

    return y_true, y_score


def plot_curves(class_names, y_true, y_score, output_path: Path, iou_thr):
    plt.figure(figsize=(10, 8))
    plotted = False
    for c, name in enumerate(class_names):
        if not y_true[c]:
            continue
        positives = int(np.sum(y_true[c]))
        negatives = len(y_true[c]) - positives
        if positives == 0:
            continue
        if negatives == 0:
            # ROC needs both classes; add one dummy negative.
            y_true[c].append(0)
            y_score[c].append(0.0)

        fpr, tpr, _ = roc_curve(y_true[c], y_score[c])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
        plotted = True

    if not plotted:
        raise ValueError("No valid classes to plot ROC.")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Object Detection ROC (object-level matching, IoU >= {iou_thr})")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot object-detection ROC curves.")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--dataset", required=True, help="Path to YOLO dataset root")
    parser.add_argument("--output", default="detection_roc_curve.png", help="Output PNG path")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    args = parser.parse_args()

    model = YOLO(args.model)
    class_names, dataset = load_dataset(Path(args.dataset))
    records = collect_predictions(model, dataset, len(class_names))
    y_true, y_score = build_roc_inputs(records, len(class_names), iou_thr=args.iou)
    plot_curves(class_names, y_true, y_score, Path(args.output), args.iou)
    print(f"Detection ROC saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
