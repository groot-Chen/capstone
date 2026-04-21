from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
)

IMG_SIZE = 224

MODEL_CONFIGS = {
    "EfficientNet-B0": {
        "checkpoint": Path(r"D:\gesture-recognition\valid_resluts\efficientnet_b0\best_efficientnet_b0.pth"),
        "model_name_in_ckpt": "efficientnet_b0",
    },
    "MobileNetV3": {
        "checkpoint": Path(r"D:\gesture-recognition\valid_resluts\mobilenetv3\best_mobilenetv3.pth"),
        "model_name_in_ckpt": "mobilenet_v3_large",
    },
    "ResNet18": {
        "checkpoint": Path(r"D:\gesture-recognition\valid_resluts\resnet18\best_resnet18.pth"),
        "model_name_in_ckpt": "resnet18",
    },
}


def build_model(model_key: str, num_classes: int) -> nn.Module:
    if model_key == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        head = model.classifier[-1]
        if not isinstance(head, nn.Linear):
            raise TypeError("EfficientNet classifier head is not nn.Linear")
        model.classifier[-1] = nn.Linear(head.in_features, num_classes)
        return model

    if model_key == "MobileNetV3":
        model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        head = model.classifier[-1]
        if not isinstance(head, nn.Linear):
            raise TypeError("MobileNetV3 classifier head is not nn.Linear")
        model.classifier[-1] = nn.Linear(head.in_features, num_classes)
        return model

    if model_key == "ResNet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        head = model.fc
        if not isinstance(head, nn.Linear):
            raise TypeError("ResNet18 fc head is not nn.Linear")
        model.fc = nn.Linear(head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model key: {model_key}")


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint format is invalid: expected dict.")

    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    if "class_names" not in ckpt:
        raise ValueError("Checkpoint does not contain 'class_names'.")

    class_names = ckpt["class_names"]
    if not isinstance(class_names, list) or not all(isinstance(x, str) for x in class_names):
        raise ValueError("class_names in checkpoint is invalid.")

    return ckpt, ckpt["model_state_dict"], class_names


@st.cache_resource
def load_predictor(model_key: str):
    config = MODEL_CONFIGS[model_key]
    checkpoint_path = config["checkpoint"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt, state_dict, class_names = load_checkpoint(checkpoint_path, device)

    model = build_model(model_key, num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_val_transform()
    return model, transform, class_names, device, ckpt


def preprocess_image(image: Image.Image, transform) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def predict(image: Image.Image, model, transform, class_names, device):
    x = preprocess_image(image, transform).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    pred_idx = int(pred_idx.item())
    confidence = float(conf.item())
    pred_name = class_names[pred_idx]

    return pred_name, confidence, probs[0]


def main():
    st.set_page_config(page_title="Gesture Recognition Demo", layout="wide")

    st.title("Static Hand Gesture Recognition Demo")
    st.caption("Upload one image, select a trained model, and view the prediction result.")

    model_key = st.sidebar.selectbox("Select Model", list(MODEL_CONFIGS.keys()))

    try:
        model, transform, class_names, device, ckpt = load_predictor(model_key)
    except Exception as e:
        st.error(f"Failed to load predictor: {e}")
        st.stop()

    st.sidebar.write("Checkpoint Path")
    st.sidebar.code(str(MODEL_CONFIGS[model_key]["checkpoint"]))

    if "image_size" in ckpt:
        st.sidebar.write(f"Image Size: {ckpt['image_size']}")
    if "best_val_acc" in ckpt:
        st.sidebar.write(f"Best Val Acc: {ckpt['best_val_acc']:.4f}")
    if "epoch" in ckpt:
        st.sidebar.write(f"Saved Epoch: {ckpt['epoch']}")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Please upload an image first.")
        return

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        pred_name, confidence, probs = predict(image, model, transform, class_names, device)

        st.subheader("Prediction Result")
        st.metric("Predicted Class", pred_name)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        topk = min(5, len(class_names))
        top_probs, top_idx = probs.topk(topk)

        st.subheader(f"Top-{topk} Predictions")
        for i in range(topk):
            idx = int(top_idx[i].item())
            prob = float(top_probs[i].item())
            label = class_names[idx]
            st.write(f"**{label}** — {prob * 100:.2f}%")
            st.progress(prob)


if __name__ == "__main__":
    main()