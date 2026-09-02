import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from ultralytics import YOLO
import yaml


def collect_samples(dataset_dir: Path):
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Expected YOLO dataset with data.yaml in: {dataset_dir}"
        )

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg.get("names", [])
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = list(names)

    test_images_dir = dataset_dir / "test" / "images"
    test_labels_dir = dataset_dir / "test" / "labels"
    if not test_images_dir.exists() or not test_labels_dir.exists():
        raise FileNotFoundError(
            f"Expected YOLO test split folders: {test_images_dir} and {test_labels_dir}"
        )

    samples = []
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for image_path in test_images_dir.iterdir():
        if image_path.suffix.lower() not in valid_ext:
            continue
        label_path = test_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            continue
        # Use first annotation class as the image label for ROC.
        true_idx = int(lines[0].split()[0])
        samples.append((image_path, true_idx))

    return samples, class_names


def infer_scores(model: YOLO, image_path: Path, num_classes: int):
    image = Image.open(image_path).convert("RGB")
    result = model(image, verbose=False)[0]
    scores = np.zeros(num_classes, dtype=np.float32)

    if result.boxes is None or len(result.boxes) == 0:
        return scores

    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for cls_idx, conf in zip(classes, confs):
        if 0 <= cls_idx < num_classes:
            scores[cls_idx] = max(scores[cls_idx], conf)

    return scores


def plot_roc_curves(y_true, y_score, class_names, output_path: Path):
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    plt.figure(figsize=(9, 7))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Emotion Detection Model")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create ROC curve for YOLO emotion model.")
    parser.add_argument("--model", required=True, help="Path to YOLO emotion model (.pt).")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to labeled dataset directory. Structure: dataset/class_name/image.jpg",
    )
    parser.add_argument(
        "--output",
        default="roc_curve.png",
        help="Output ROC image path (default: roc_curve.png).",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    samples, class_names = collect_samples(dataset_dir)
    if not samples:
        raise ValueError("No images found in dataset folder.")
    if len(class_names) < 2:
        raise ValueError("ROC requires at least 2 classes.")

    model = YOLO(str(model_path))

    y_true = []
    y_score = []

    for image_path, true_idx in samples:
        y_true.append(true_idx)
        y_score.append(infer_scores(model, image_path, len(class_names)))

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    plot_roc_curves(y_true, y_score, class_names, output_path)
    print(f"ROC curve saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
