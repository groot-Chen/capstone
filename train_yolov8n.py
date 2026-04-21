from pathlib import Path
from ultralytics.models.yolo.model import YOLO 
import torch

DATA_YAML = r"D:\gesture-recognition\yolo_train_dataset\data.yaml"
PROJECT_DIR = r"D:\gesture-recognition\yolo\runs"
RUN_NAME = "gesture_train"

MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
IMGSZ = 640
BATCH = 16
WORKERS = 4
PATIENCE = 15

SEED = 42
SAVE_PERIOD = -1
PRETRAINED = True


def check_paths():
    data_yaml_path = Path(DATA_YAML)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    dataset_root = data_yaml_path.parent
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Required directory not found: {d}")

    print("Dataset structure check passed.")
    print(f"data.yaml: {data_yaml_path}")


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("CUDA is not available. Training will use CPU.")
        return "cpu"


def main():
    check_paths()
    device = get_device()

    print("\n========== Training Configuration ==========")
    print(f"Device      : {device}")
    print(f"Data YAML   : {DATA_YAML}")
    print(f"Model       : {MODEL_NAME}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Image Size  : {IMGSZ}")
    print(f"Batch Size  : {BATCH}")
    print(f"Workers     : {WORKERS}")
    print(f"Patience    : {PATIENCE}")
    print(f"Project Dir : {PROJECT_DIR}")
    print(f"Run Name    : {RUN_NAME}")
    print("===========================================\n")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        patience=PATIENCE,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        pretrained=PRETRAINED,
        optimizer="auto",
        seed=SEED,
        save=True,
        save_period=SAVE_PERIOD,
        val=True,
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    run_dir = Path(PROJECT_DIR) / RUN_NAME
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    results_png = run_dir / "results.png"
    results_csv = run_dir / "results.csv"
    pr_curve = run_dir / "PR_curve.png"
    f1_curve = run_dir / "F1_curve.png"
    confusion_matrix = run_dir / "confusion_matrix.png"
    confusion_matrix_norm = run_dir / "confusion_matrix_normalized.png"

    print("\n========== Training Finished ==========")
    print(f"Run directory: {run_dir}")

    if best_pt.exists():
        print(f"Best model   : {best_pt}")
    else:
        print("Best model   : not found")

    if last_pt.exists():
        print(f"Last model   : {last_pt}")
    else:
        print("Last model   : not found")

    if results_png.exists():
        print(f"Results plot : {results_png}")
    if results_csv.exists():
        print(f"Results csv  : {results_csv}")
    if pr_curve.exists():
        print(f"PR curve     : {pr_curve}")
    if f1_curve.exists():
        print(f"F1 curve     : {f1_curve}")
    if confusion_matrix.exists():
        print(f"Conf matrix  : {confusion_matrix}")
    if confusion_matrix_norm.exists():
        print(f"Norm matrix  : {confusion_matrix_norm}")

    print("=======================================\n")


if __name__ == "__main__":
    main()