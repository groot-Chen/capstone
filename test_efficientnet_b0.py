# ===== Test Summary =====
# test_loss: 0.266826
# accuracy: 0.928078
# precision_macro: 0.921761
# recall_macro: 0.928701
# f1_macro: 0.924949
# precision_weighted: 0.928752
# recall_weighted: 0.928078
# f1_weighted: 0.928145

# ===== Per-class Metrics =====
#     class_name  precision    recall  f1_score  support
# 0         call   0.902741  0.943623  0.922729     1082
# 1      dislike   0.945860  0.970588  0.958065     1224
# 2         fist   0.936740  0.945946  0.941320     1221
# 3         four   0.865979  0.897288  0.881356     1217
# 4   hand_heart   0.997503  0.993369  0.995432     2413
# 5         like   0.925129  0.889165  0.906790     1209
# 6   no_gesture   0.914070  0.968000  0.940262     1000
# 7           ok   0.896664  0.913765  0.905133     1206
# 8          one   0.839342  0.867909  0.853386     1234
# 9        peace   0.936810  0.874227  0.904437     2425
# 10       point   0.955041  0.955692  0.955366     1467
# 11        stop   0.965298  0.944905  0.954993     2414
# 12       three   0.935442  0.916814  0.926034     3967
# 13      two_up   0.888032  0.920521  0.903985     2378

# All results saved to: D:\gesture-recognition\resluts\efficientnet_b0
import os
import json
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


TEST_DIR = r"D:\gesture-recognition\test_dataset"
CHECKPOINT_PATH = r"D:\gesture-recognition\valid_resluts\efficientnet_b0\best_efficientnet_b0.pth"
OUTPUT_DIR = r"D:\gesture-recognition\testing_resluts\efficientnet_b0"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
FORCE_CPU = False

# sklearn stubs sometimes type zero_division too narrowly; int 0 is valid at runtime.
SK_ZERO_DIV: Any = 0


class ImageFolderWithPaths(Dataset):
    """Wrap ImageFolder and also return image paths without overriding base signatures."""

    def __init__(self, root: str, transform=None):
        self._folder = datasets.ImageFolder(root, transform=transform)

    def __len__(self) -> int:
        return len(self._folder)

    def __getitem__(self, index: int):
        image, label = self._folder[index]
        path, _ = self._folder.samples[index]
        return image, label, path

    @property
    def classes(self):
        return self._folder.classes


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    head = model.classifier[-1]
    if not isinstance(head, nn.Linear):
        raise TypeError("Expected EfficientNet classifier final layer to be nn.Linear")
    in_features = head.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_test_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, class_names):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Testing", ncols=100)

    for images, labels, paths in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(list(paths))

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataloader.dataset)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=SK_ZERO_DIV
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=SK_ZERO_DIV
    )

    p_cls, r_cls, f1_cls, support_cls = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=SK_ZERO_DIV
    )

    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=SK_ZERO_DIV,
    )

    metrics = {
        "test_loss": float(avg_loss),
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    per_class_df = pd.DataFrame({
        "class_name": class_names,
        "precision": p_cls,
        "recall": r_cls,
        "f1_score": f1_cls,
        "support": support_cls
    })

    predictions_df = pd.DataFrame({
        "image_path": all_paths,
        "true_label_idx": all_labels,
        "pred_label_idx": all_preds,
        "true_label_name": [class_names[i] for i in all_labels],
        "pred_label_name": [class_names[i] for i in all_preds],
        "correct": (all_labels == all_preds)
    })

    for i, cls_name in enumerate(class_names):
        predictions_df[f"prob_{cls_name}"] = all_probs[:, i]

    return metrics, per_class_df, cls_report, predictions_df, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiclass_roc(y_true, y_prob, class_names, save_path):
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiclass_pr(y_true, y_prob, class_names, save_path):
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        if y_true_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=f"{class_name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-class Precision-Recall Curves")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_results(output_dir, metrics, per_class_df, cls_report, predictions_df, y_true, y_pred, y_prob, class_names):
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "summary_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    per_class_df.to_csv(
        os.path.join(output_dir, "per_class_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(cls_report)

    predictions_df.to_csv(
        os.path.join(output_dir, "predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    mis_df = predictions_df[predictions_df["correct"] == False].copy()
    mis_df.to_csv(
        os.path.join(output_dir, "misclassified_samples.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix_normalized.png"), normalize=True)
    plot_multiclass_roc(y_true, y_prob, class_names, os.path.join(output_dir, "roc_curves.png"))
    plot_multiclass_pr(y_true, y_prob, class_names, os.path.join(output_dir, "pr_curves.png"))

    with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test set: {TEST_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    test_transform = get_test_transform(IMG_SIZE)
    test_dataset = ImageFolderWithPaths(TEST_DIR, transform=test_transform)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes:")
    print(class_names)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    model = build_model(num_classes=num_classes)
    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    metrics, per_class_df, cls_report, predictions_df, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, device, class_names
    )

    save_results(
        output_dir=OUTPUT_DIR,
        metrics=metrics,
        per_class_df=per_class_df,
        cls_report=cls_report,
        predictions_df=predictions_df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names
    )

    print("\n===== Test Summary =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n===== Per-class Metrics =====")
    print(per_class_df)
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()