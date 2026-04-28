import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Constants ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


# ── Albumentations Transforms ──────────────────────────────────────────────
def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.GaussNoise(p=1.0),
                ],
                p=0.2,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# ── Dataset Class ──────────────────────────────────────────────────────────
class PlantDataset(Dataset):
    def __init__(self, image_paths: list, labels: list, transform: A.Compose = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = np.array(
            Image.open(self.image_paths[idx]).convert("RGB")
        )  # Albumentations expects numpy HWC
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]


# ── Data Collection
def collect_data(base_path: str) -> tuple[list, list, list, dict]:
    """Scan directory and return image paths, labels, class names, class_to_idx."""
    class_names = sorted(
        [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    )
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    all_images, all_labels = [], []
    for cls in class_names:
        cls_path = os.path.join(base_path, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(cls_path, fname))
                all_labels.append(class_to_idx[cls])

    return all_images, all_labels, class_names, class_to_idx


# ── Sampler ────────────────────────────────────────────────────────────────
def build_sampler(labels: list) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ── Main Factory ───────────────────────────────────────────────────────────
def get_dataloaders(
    base_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, list, dict]:
    """
    Returns: train_loader, val_loader, test_loader, class_names, class_to_idx
    """
    all_images, all_labels, class_names, class_to_idx = collect_data(base_path)

    # Stratified 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=0.30, stratify=all_labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_ds = PlantDataset(X_train, y_train, get_train_transforms())
    val_ds = PlantDataset(X_val, y_val, get_val_transforms())
    test_ds = PlantDataset(X_test, y_test, get_val_transforms())

    # Windows guard
    safe_workers = 0 if os.name == "nt" else num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=build_sampler(y_train),
        num_workers=safe_workers,
        pin_memory=True,
        persistent_workers=safe_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=True,
        persistent_workers=safe_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=True,
        persistent_workers=safe_workers > 0,
    )

    return train_loader, val_loader, test_loader, class_names, class_to_idx


if __name__ == "__main__":
    BASE = "data/raw/PlantVillage/PlantVillage"
    print(f" BASE is {BASE}")
    train_loader, val_loader, test_loader, class_names, _ = get_dataloaders(BASE)

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape : {imgs.shape}")  # [32, 3, 224, 224]
    print(f"Label shape : {labels.shape}")  # [32]
    print(f"Dtype       : {imgs.dtype}")  # torch.float32
    print(f"Num classes : {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f" {i:2d}: {name}")
    print("dataset.py ✅")
