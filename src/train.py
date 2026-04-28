import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model


# ── Config ───────────────────────────────────────────────────────────────
BASE = "data/raw/PlantVillage/PlantVillage"
NUM_CLASSES = 15
EPOCHS_HEAD = 10  # Phase 1: train head only
EPOCHS_FULL = 10  # Phase 2: fine-tune full network
LR_HEAD = 1e-3
LR_FULL = 1e-4
BATCH_SIZE = 32
CHECKPOINT = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


def save_checkpoint(model, path, epoch, val_acc):
    torch.save({"epoch": epoch, "val_acc": val_acc, "state": model.state_dict()}, path)
    print(f"  ✅ Saved checkpoint (val_acc={val_acc:.4f})")


def train_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    best_acc,
):
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, CHECKPOINT, epoch, val_acc)

    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, class_names, _ = get_dataloaders(
        BASE, batch_size=BATCH_SIZE
    )

    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Head only ────────────────────────────────────────────────
    print("\n── Phase 1: Training head only ──")
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)

    best_acc = train_phase(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        EPOCHS_HEAD,
        DEVICE,
        best_acc=0.0,
    )

    # ── Phase 2: Full fine-tune ───────────────────────────────────────────
    print("\n── Phase 2: Fine-tuning full network ──")
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = Adam(model.parameters(), lr=LR_FULL)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL)

    best_acc = train_phase(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        EPOCHS_FULL,
        DEVICE,
        best_acc,
    )

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT}")
