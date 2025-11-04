import torch
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

MODEL_PATH = "../ml_model/skin_cancer_model.pth"

app = FastAPI()

# Configure CORS middleware (Put this right after app initialization)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load class labels (must match your original dataset)
class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Load the trained model
class SkinCancerResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(SkinCancerResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  # Set to False to avoid downloading weights
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
# Initialize model and load weights
# Initialize model and load weights (with key remapping)
num_classes = len(class_labels)
model = SkinCancerResNet(num_classes)

# Load checkpoint
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

# Remap keys if they start with "model." instead of "resnet."
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_state_dict[k.replace("model.", "resnet.")] = v
    else:
        new_state_dict[k] = v

# Load non-strict to allow for any harmless mismatches
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Image transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.get("/")
def root():
    return {"message": "AI backend running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = float(probabilities[predicted_class])

    predicted_label = class_labels[predicted_class]
    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4)
    }
