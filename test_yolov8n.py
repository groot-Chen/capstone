from pathlib import Path
from ultralytics.models.yolo.model import YOLO 
import torch
import json
import csv
import shutil


# =========================================================
# Config
# =========================================================
MODEL_PATH = r"D:\gesture-recognition\yolo\runs\gesture_train\weights\best.pt"

TEST_DATA_YAML = r"D:\gesture-recognition\yolo_test_dataset\data.yaml"
TEST_IMAGE_DIR = r"D:\gesture-recognition\yolo_test_dataset\images\test"
TEST_LABEL_DIR = r"D:\gesture-recognition\yolo_test_dataset\labels\test"

OUTPUT_ROOT = r"D:\gesture-recognition\testing_resluts\yolov8n"
RUN_NAME = "gesture_test"

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.60
DEVICE = 0 if torch.cuda.is_available() else "cpu"

SAVE_PRED_IMAGES = True
MAX_VIS_IMAGES = None  # Set to an integer like 200 if you do not want to copy all images


# =========================================================
# Utility functions
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def extract_class_names(model):
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


def count_files_in_dir(folder: Path, suffixes=None):
    if not folder.exists():
        return 0
    if suffixes is None:
        return len([p for p in folder.iterdir() if p.is_file()])
    return len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes])


def save_summary_txt(save_dir: Path, metrics, class_names, num_test_images, num_test_labels):
    lines = []
    lines.append("YOLO Test Summary")
    lines.append("=" * 70)
    lines.append(f"Model Path      : {MODEL_PATH}")
    lines.append(f"Test Data YAML  : {TEST_DATA_YAML}")
    lines.append(f"Test Image Dir  : {TEST_IMAGE_DIR}")
    lines.append(f"Test Label Dir  : {TEST_LABEL_DIR}")
    lines.append(f"Number of Images: {num_test_images}")
    lines.append(f"Number of Labels: {num_test_labels}")
    lines.append("")

    lines.append("Overall Metrics")
    lines.append("-" * 70)
    lines.append(f"Precision : {safe_float(metrics.box.mp):.6f}")
    lines.append(f"Recall    : {safe_float(metrics.box.mr):.6f}")
    lines.append(f"mAP50     : {safe_float(metrics.box.map50):.6f}")
    lines.append(f"mAP50-95  : {safe_float(metrics.box.map):.6f}")
    lines.append("")

    ap50_list = metrics.box.ap50
    ap_list = metrics.box.ap
    p_list = metrics.box.p
    r_list = metrics.box.r

    lines.append("Per-class Metrics")
    lines.append("-" * 70)
    for i, name in enumerate(class_names):
        p = safe_float(p_list[i]) if i < len(p_list) else None
        r = safe_float(r_list[i]) if i < len(r_list) else None
        ap50 = safe_float(ap50_list[i]) if i < len(ap50_list) else None
        ap = safe_float(ap_list[i]) if i < len(ap_list) else None

        lines.append(
            f"{i:02d} | {name:<12} | "
            f"P={p:.6f} | R={r:.6f} | mAP50={ap50:.6f} | mAP50-95={ap:.6f}"
        )

    lines.append("")
    lines.append("Expected Output Files")
    lines.append("-" * 70)
    lines.append("summary.txt")
    lines.append("thesis_metrics.csv")
    lines.append("thesis_metrics.json")
    lines.append("confusion_matrix.png")
    lines.append("confusion_matrix_normalized.png")
    lines.append("PR_curve.png")
    lines.append("P_curve.png")
    lines.append("R_curve.png")
    lines.append("F1_curve.png")
    lines.append("prediction_images/")

    write_text(save_dir / "summary.txt", "\n".join(lines))


def save_metrics_csv(save_dir: Path, metrics, class_names):
    csv_path = save_dir / "thesis_metrics.csv"

    ap50_list = metrics.box.ap50
    ap_list = metrics.box.ap
    p_list = metrics.box.p
    r_list = metrics.box.r

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "section",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "mAP50",
            "mAP50_95"
        ])

        writer.writerow([
            "overall",
            "",
            "all",
            safe_float(metrics.box.mp),
            safe_float(metrics.box.mr),
            safe_float(metrics.box.map50),
            safe_float(metrics.box.map),
        ])

        for i, name in enumerate(class_names):
            writer.writerow([
                "class",
                i,
                name,
                safe_float(p_list[i]) if i < len(p_list) else None,
                safe_float(r_list[i]) if i < len(r_list) else None,
                safe_float(ap50_list[i]) if i < len(ap50_list) else None,
                safe_float(ap_list[i]) if i < len(ap_list) else None,
            ])


def save_metrics_json(save_dir: Path, metrics, class_names, num_test_images, num_test_labels):
    ap50_list = metrics.box.ap50
    ap_list = metrics.box.ap
    p_list = metrics.box.p
    r_list = metrics.box.r

    result = {
        "model_path": MODEL_PATH,
        "test_data_yaml": TEST_DATA_YAML,
        "test_image_dir": TEST_IMAGE_DIR,
        "test_label_dir": TEST_LABEL_DIR,
        "num_test_images": num_test_images,
        "num_test_labels": num_test_labels,
        "overall": {
            "precision": safe_float(metrics.box.mp),
            "recall": safe_float(metrics.box.mr),
            "mAP50": safe_float(metrics.box.map50),
            "mAP50_95": safe_float(metrics.box.map),
        },
        "per_class": []
    }

    for i, name in enumerate(class_names):
        result["per_class"].append({
            "class_id": i,
            "class_name": name,
            "precision": safe_float(p_list[i]) if i < len(p_list) else None,
            "recall": safe_float(r_list[i]) if i < len(r_list) else None,
            "mAP50": safe_float(ap50_list[i]) if i < len(ap50_list) else None,
            "mAP50_95": safe_float(ap_list[i]) if i < len(ap_list) else None,
        })

    with open(save_dir / "thesis_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def copy_prediction_images(src_dir: Path, dst_dir: Path, max_images=None):
    if not src_dir.exists():
        print(f"[Warning] Prediction source directory not found: {src_dir}")
        return

    ensure_dir(dst_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in src_dir.iterdir() if p.suffix.lower() in image_exts]
    images = sorted(images)

    if max_images is not None:
        images = images[:max_images]

    for img_path in images:
        shutil.copy2(img_path, dst_dir / img_path.name)

    print(f"[Info] Copied {len(images)} visualized prediction images to: {dst_dir}")


# =========================================================
# Main
# =========================================================
def main():
    model_path = Path(MODEL_PATH)
    test_data_yaml_path = Path(TEST_DATA_YAML)
    test_image_dir_path = Path(TEST_IMAGE_DIR)
    test_label_dir_path = Path(TEST_LABEL_DIR)

    output_root = Path(OUTPUT_ROOT)
    save_dir = output_root / RUN_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not test_data_yaml_path.exists():
        raise FileNotFoundError(f"Test data.yaml not found: {test_data_yaml_path}")

    if not test_image_dir_path.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_image_dir_path}")

    if not test_label_dir_path.exists():
        raise FileNotFoundError(f"Test label directory not found: {test_label_dir_path}")

    ensure_dir(save_dir)

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    label_suffixes = {".txt"}

    num_test_images = count_files_in_dir(test_image_dir_path, image_suffixes)
    num_test_labels = count_files_in_dir(test_label_dir_path, label_suffixes)

    print("=" * 80)
    print("YOLO Test Evaluation for Thesis")
    print("=" * 80)
    print(f"Model Path      : {model_path}")
    print(f"Test Data YAML  : {test_data_yaml_path}")
    print(f"Test Image Dir  : {test_image_dir_path}")
    print(f"Test Label Dir  : {test_label_dir_path}")
    print(f"Number of Images: {num_test_images}")
    print(f"Number of Labels: {num_test_labels}")
    print(f"Output Dir      : {save_dir}")
    print(f"Device          : {DEVICE}")
    print("=" * 80)

    model = YOLO(str(model_path))
    class_names = extract_class_names(model)

    # -----------------------------------------------------
    # 1) Official evaluation on the test set
    # -----------------------------------------------------
    print("\n[1/2] Running official evaluation on the test set...")
    metrics = model.val(
        data=str(test_data_yaml_path),
        split="test",
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        project=str(output_root),
        name=RUN_NAME,
        exist_ok=True,
        save_json=True,
        plots=True,
        verbose=True
    )

    print("\n[Info] Test evaluation finished.")
    print(f"Precision : {metrics.box.mp:.6f}")
    print(f"Recall    : {metrics.box.mr:.6f}")
    print(f"mAP50     : {metrics.box.map50:.6f}")
    print(f"mAP50-95  : {metrics.box.map:.6f}")

    save_summary_txt(save_dir, metrics, class_names, num_test_images, num_test_labels)
    save_metrics_csv(save_dir, metrics, class_names)
    save_metrics_json(save_dir, metrics, class_names, num_test_images, num_test_labels)

    # -----------------------------------------------------
    # 2) Save visualized prediction images
    # -----------------------------------------------------
    if SAVE_PRED_IMAGES:
        print("\n[2/2] Running prediction on test images for visualization...")

        predict_name = f"{RUN_NAME}_predictions"
        model.predict(
            source=str(test_image_dir_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=DEVICE,
            save=True,
            project=str(output_root),
            name=predict_name,
            exist_ok=True,
            verbose=True
        )

        pred_src_dir = output_root / predict_name
        pred_dst_dir = save_dir / "prediction_images"
        copy_prediction_images(pred_src_dir, pred_dst_dir, MAX_VIS_IMAGES)

    print("\n" + "=" * 80)
    print("All testing work finished.")
    print(f"Main result folder: {save_dir}")
    print("Important files:")
    print(f"- {save_dir / 'summary.txt'}")
    print(f"- {save_dir / 'thesis_metrics.csv'}")
    print(f"- {save_dir / 'thesis_metrics.json'}")
    print(f"- {save_dir / 'prediction_images'}")
    print("=" * 80)


if __name__ == "__main__":
    main()