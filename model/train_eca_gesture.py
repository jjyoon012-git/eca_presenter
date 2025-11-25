import os
import math
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# ==========================
# 1. 하이퍼파라미터 & 경로
# ==========================

NUM_CLASSES = 4  # ok, fist, index_up, v_sign

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"

ROOT_DIR = Path(__file__).resolve().parent.parent  # eca_presenter/
DATA_DIR = ROOT_DIR / "data"
SAVE_PATH = ROOT_DIR / "model" / "eca_gesture.pth"


# ==========================
# 2. ECA 블록 정의
# ==========================

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) 블록
    """

    def __init__(self, channels: int, gamma: float = 2, b: float = 1):
        super().__init__()
        t = int(abs((math.log2(channels) / gamma) + b))
        k = t if t % 2 == 1 else t + 1
        k = max(1, k)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=k, padding=(k - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                 # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)                     # (B, 1, C)
        y = self.sigmoid(y)                  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


# ==========================
# 3. ECA-GestureNet 모델 정의
# ==========================

class ECAGestureNet(nn.Module):
    """
    간단한 CNN + ECA 구조 (224x224 RGB 입력)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x224x224 -> 32x112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ECABlock(32),

            # Block 2: 32x112x112 -> 64x56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ECABlock(64),

            # Block 3: 64x56x56 -> 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ECABlock(128),

            # Block 4: 128x28x28 -> 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ECABlock(256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==========================
# 4. 데이터셋 / DataLoader
# ==========================

def get_dataloaders(data_dir: Path,
                    batch_size: int = BATCH_SIZE
                    ) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_tf = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_set = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    val_set = datasets.ImageFolder(root=str(val_dir), transform=val_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )

    idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}
    print("class_to_idx:", train_set.class_to_idx)
    print("train 개수:", len(train_set), " / val 개수:", len(val_set))

    return train_loader, val_loader, idx_to_class


# ==========================
# 5. 학습 / 검증 루프
# ==========================

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for step, (images, labels) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 20 == 0:
            print(f"  [Train] Epoch {epoch} Step {step+1}/{len(loader)} "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"==> [Train] Epoch {epoch} | Loss: {epoch_loss:.4f}, "
          f"Acc: {epoch_acc*100:.2f}%")
    return epoch_loss


def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"==> [Val]   Epoch {epoch} | Loss: {epoch_loss:.4f}, "
          f"Acc: {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc


# ==========================
# 6. 메인 학습 함수
# ==========================

def main():
    print("===== ECA-GestureNet 학습 시작 =====")
    print(f"DEVICE: {DEVICE}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"모델 저장 경로: {SAVE_PATH}")

    os.makedirs(SAVE_PATH.parent, exist_ok=True)

    train_loader, val_loader, idx_to_class = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print("idx_to_class:", idx_to_class)

    model = ECAGestureNet(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"*** Best model updated (val_acc={val_acc*100:.2f}%) "
                  f"→ {SAVE_PATH}")

    print("\n===== 학습 종료 =====")
    print(f"Best Val Acc: {best_val_acc*100:.2f}%")

    # labels.txt 저장
    labels_txt_path = ROOT_DIR / "assets" / "labels.txt"
    os.makedirs(labels_txt_path.parent, exist_ok=True)
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    with open(labels_txt_path, "w", encoding="utf-8") as f:
        for name in labels:
            f.write(name + "\n")
    print(f"클래스 라벨을 {labels_txt_path}에 저장했습니다.")


if __name__ == "__main__":
    main()