import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────
# Configuration
# ─────────────────────────────
DATA_PATH = "HAM10000"
IMAGE_PATH = os.path.join(DATA_PATH, "images")
METADATA_PATH = os.path.join(DATA_PATH, "HAM10000_metadata.csv")
MODEL_PATH = "skin_cancer_model.pth"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────
# Dataset
# ─────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, image_dir, metadata, transform=None):
        self.image_dir = image_dir
        self.metadata = metadata
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(metadata["dx"].unique())}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")
        image = Image.open(image_path).convert("RGB")
        label = self.label_map[row["dx"]]
        if self.transform:
            image = self.transform(image)
        return image, label, row["image_id"]

# ─────────────────────────────
# Model
# ─────────────────────────────
class SkinCancerResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ─────────────────────────────
# Train function
# ─────────────────────────────
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")
    print("✅ Training complete")

# ─────────────────────────────
# Main
# ─────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting training...")

    metadata = pd.read_csv(METADATA_PATH)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SkinDataset(IMAGE_PATH, metadata, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(metadata["dx"].unique())
    model = SkinCancerResNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, dataloader, criterion, optimizer, EPOCHS)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
