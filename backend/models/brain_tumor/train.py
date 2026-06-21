import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import BrainTumorModel
from PIL import Image
import os

# Config
TRAIN_PATH = "../../datasets/brain_tumor/Training"
TEST_PATH = "../../datasets/brain_tumor/Testing"
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0001
NUM_CLASSES = 4
SAVE_PATH = "brain_tumor_model.pth"

# Separate transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load train and test SEPARATELY — this is the key fix
train_dataset = ImageFolder(root=TRAIN_PATH, transform=train_transform)
val_dataset = ImageFolder(root=TEST_PATH, transform=val_transform)

print(f"Classes: {train_dataset.classes}")
print(f"Train images: {len(train_dataset)}")
print(f"Val images: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_size = len(train_dataset)
val_size = len(val_dataset)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Model — freeze base, train classifier first
model = BrainTumorModel(num_classes=NUM_CLASSES).to(device)

for param in model.base_model.parameters():
    param.requires_grad = False
for param in model.base_model.classifier.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Unfreeze all layers at epoch 5
    if epoch == 5:
        print("  🔓 Unfreezing all layers...")
        for param in model.base_model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR / 5, weight_decay=1e-4
        )

    # Train
    model.train()
    train_loss, train_correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # Validate
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / train_size * 100
    val_acc = val_correct / val_size * 100
    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")

print(f"\n🎉 Training complete! Best Val Accuracy: {best_val_acc:.2f}%")