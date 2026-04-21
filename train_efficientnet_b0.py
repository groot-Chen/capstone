# Training finished. Best Val Acc: 0.9361
# Best Epoch: 18
# Total Time: 2946.2s
import importlib
import json
import random
import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms
    from torchvision.models import EfficientNet_B0_Weights

    tqdm = importlib.import_module("tqdm").tqdm
    plt = importlib.import_module("matplotlib.pyplot")
    import numpy as np
except ModuleNotFoundError as exc:
    print(f"Missing dependency: {exc}")
    print("Please install training dependencies first:")
    print(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m pip install torch torchvision pillow matplotlib tqdm")
    sys.exit(1)

DATA_DIR = Path(r"D:\gesture-recognition\train_dataset")
OUTPUT_DIR = Path(r"D:\gesture-recognition\valid_resluts\efficientnet_b0")
BEST_MODEL_PATH = OUTPUT_DIR / "best_efficientnet_b0.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "last_efficientnet_b0.pth"
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"
TRAIN_LOG_PATH = OUTPUT_DIR / "train_log.json"
CURVES_PATH = OUTPUT_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_JSON_PATH = OUTPUT_DIR / "confusion_matrix.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


def build_dataloaders():
    train_transform, val_transform = build_transforms()

    base_dataset = datasets.ImageFolder(DATA_DIR)
    if len(base_dataset) == 0:
        raise ValueError(f"No images found in dataset directory: {DATA_DIR}")

    class_names = base_dataset.classes
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes for training, but found: {num_classes}")

    train_size = int(TRAIN_RATIO * len(base_dataset))
    val_size = len(base_dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Invalid split: train_size={train_size}, val_size={val_size}, total={len(base_dataset)}"
        )

    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    shuffled_indices = torch.randperm(
        len(base_dataset),
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    ).tolist()
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return base_dataset, class_names, train_loader, val_loader


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    classifier_head = model.classifier[-1]
    if not isinstance(classifier_head, nn.Linear):
        raise TypeError("EfficientNet classifier head is not nn.Linear")
    in_features = classifier_head.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress_bar = tqdm(
        loader,
        desc=f"Train {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

        current_loss = running_loss / running_total
        current_acc = running_correct / running_total

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.4f}",
        })

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress_bar = tqdm(
        loader,
        desc=f"Val   {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            current_loss = running_loss / running_total
            current_acc = running_correct / running_total

            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}",
            })

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


def collect_predictions(model, loader):
    model.eval()
    all_labels = []
    all_preds = []

    progress_bar = tqdm(
        loader,
        desc="Confusion Matrix",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_val_acc, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "model_name": "efficientnet_b0",
        },
        path,
    )


def plot_training_curves(history, save_path: Path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_image(cm, class_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_confusion_matrix(y_true, y_pred, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        cm[int(true_idx), int(pred_idx)] += 1
    return cm


def main():
    print(f"Using device: {device}")

    if not DATA_DIR.exists():
        print(f"Dataset directory does not exist: {DATA_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        base_dataset, class_names, train_loader, val_loader = build_dataloaders()
    except Exception as exc:
        print(f"Dataset error: {exc}")
        return

    num_classes = len(class_names)

    print(f"Found {len(base_dataset)} images")
    print(f"Found {num_classes} classes")
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    save_json(class_names, CLASS_NAMES_PATH)

    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_acc = 0.0
    best_epoch = -1
    no_improve_epochs = 0
    history = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch + 1, NUM_EPOCHS
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, epoch + 1, NUM_EPOCHS
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"LR: {current_lr:.6f} "
            f"Time: {epoch_time:.1f}s"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
            }
        )

        save_checkpoint(
            LAST_MODEL_PATH,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_val_acc,
            class_names,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve_epochs = 0

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_val_acc,
                class_names,
            )
            print(f"Best model saved to: {BEST_MODEL_PATH}")
        else:
            no_improve_epochs += 1

        save_json(
            {
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "history": history,
                "model_name": "efficientnet_b0",
            },
            TRAIN_LOG_PATH,
        )

        plot_training_curves(history, CURVES_PATH)

        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    total_time = time.time() - start_time

    best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    all_labels, all_preds = collect_predictions(model, val_loader)
    cm = compute_confusion_matrix(all_labels, all_preds, num_classes=len(class_names))

    plot_confusion_matrix_image(cm, class_names, CONFUSION_MATRIX_PATH)

    save_json(
        {
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
            "model_name": "efficientnet_b0",
        },
        CONFUSION_MATRIX_JSON_PATH,
    )

    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Best model path: {BEST_MODEL_PATH}")
    print(f"Last model path: {LAST_MODEL_PATH}")
    print(f"Class names path: {CLASS_NAMES_PATH}")
    print(f"Train log path: {TRAIN_LOG_PATH}")
    print(f"Training curves path: {CURVES_PATH}")
    print(f"Confusion matrix image path: {CONFUSION_MATRIX_PATH}")
    print(f"Confusion matrix json path: {CONFUSION_MATRIX_JSON_PATH}")


if __name__ == "__main__":
    main()