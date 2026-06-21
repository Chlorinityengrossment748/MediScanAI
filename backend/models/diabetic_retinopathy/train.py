import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import DiabeticRetinopathyModel
import pandas as pd
import os

# Config
CSV_PATH = "../../datasets/diabetic_retinopathy/aptos2019-blindness-detection/train.csv"
IMG_DIR = "../../datasets/diabetic_retinopathy/aptos2019-blindness-detection/train_images"
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0001
NUM_CLASSES = 5
SAVE_PATH = "diabetic_retinopathy_model.pth"

# Custom Dataset for APTOS
class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
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

# Load full dataset
full_dataset = APTOSDataset(CSV_PATH, IMG_DIR, transform=train_transform)
print(f"Total images: {len(full_dataset)}")

# Split 80/20
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Apply val_transform to validation set
val_data.dataset.transform = val_transform

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Model — freeze base layers first
model = DiabeticRetinopathyModel(num_classes=NUM_CLASSES).to(device)

for param in model.base_model.parameters():
    param.requires_grad = False
for param in model.base_model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Unfreeze all layers after epoch 5
    if epoch == 5:
        print("  🔓 Unfreezing all layers for fine-tuning...")
        for param in model.base_model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR/10, weight_decay=1e-4
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
          f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")

print(f"\n🎉 Training complete! Best Val Accuracy: {best_val_acc:.2f}%")