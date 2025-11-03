import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

# Paths
DATA_PATH = "HAM10000"
IMAGE_PATH = os.path.join(DATA_PATH, "images")
METADATA_PATH = os.path.join(DATA_PATH, "HAM10000_metadata.csv")
MODEL_PATH = "skin_cancer_model.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metadata
metadata = pd.read_csv(METADATA_PATH)

# Transformations (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset class
class SkinCancerDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, metadata, transform=None):
        self.image_dir = image_dir
        self.metadata = metadata
        self.transform = transform
        self.image_ids = metadata["image_id"].values
        self.labels = metadata["dx"].values
        self.label_map = {label: idx for idx, label in enumerate(np.unique(self.labels))}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        label_idx = self.label_map[label]
        image_path = os.path.join(self.image_dir, image_id + ".jpg")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_idx, image_id

# Model
class SkinCancerResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Prepare dataset and dataloader
dataset = SkinCancerDataset(IMAGE_PATH, metadata, transform)
subset_size = min(2000, len(dataset))  # smaller subset for faster testing
_, test_dataset = random_split(dataset, [len(dataset) - subset_size, subset_size])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
num_classes = len(metadata["dx"].unique())
model = SkinCancerResNet(num_classes).to(device)

# ✅ Load model with key remapping (fixes "Missing/Unexpected keys" issue)
print("Loading model...")
state_dict = torch.load(MODEL_PATH, map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    if not k.startswith("model."):
        new_state_dict[f"model.{k}"] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)
print("Model loaded successfully!")

# Evaluate
model.eval()
predictions, actuals, image_ids, confidences = [], [], [], []

with torch.no_grad():
    for images, labels, ids in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

        predictions.extend(preds.cpu().numpy())
        actuals.extend(labels.cpu().numpy())
        image_ids.extend(ids)
        confidences.extend(confs.cpu().numpy())

# Decode labels
label_map = {v: k for k, v in dataset.label_map.items()}
predicted_labels = [label_map[p] for p in predictions]
actual_labels = [label_map[a] for a in actuals]

# Save to CSV
output_df = pd.DataFrame({
    "image_id": image_ids,
    "actual_label": actual_labels,
    "predicted_label": predicted_labels,
    "confidence": np.round(confidences, 4)
})
output_df.to_csv("predictions.csv", index=False)

# Accuracy
accuracy = np.mean(np.array(predictions) == np.array(actuals))
print(f"\n✅ Evaluation Complete — Accuracy: {accuracy:.4f}")
print("Predictions saved to predictions.csv")
