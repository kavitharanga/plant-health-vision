import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders
from model import build_model

# ── Config (must match train.py exactly) ─────────────────────────────────
BASE = "data/raw/PlantVillage/PlantVillage"
NUM_CLASSES = 15
BATCH_SIZE = 32
CHECKPOINT = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load model ────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, num_classes: int) -> nn.Module:
    model = build_model(num_classes=num_classes, freeze_backbone=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}")
    return model


# ── Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader, class_names: list):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    print("── Classification Report ──────────────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved → confusion_matrix.png")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Recreate the exact same split (random_state=42 matches dataset.py)
    _, _, test_loader, class_names, _ = get_dataloaders(BASE, batch_size=BATCH_SIZE)

    model = load_model(CHECKPOINT, NUM_CLASSES)
    evaluate(model, test_loader, class_names)
