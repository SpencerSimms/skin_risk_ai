import os
from supabase import create_client, Client
from dotenv import load_dotenv
import uuid
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
import time

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use the service role key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.get("/")
def root():
    return {"message": "AI backend running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: str = Form(...)):
    # Read the image into memory
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Transform for prediction
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = float(probabilities[predicted_class])

    predicted_label = class_labels[predicted_class]

    # --- Upload to Supabase storage ---
    timestamp = int(time.time())
    file_name = f"{user_id}_{timestamp}_{file.filename}"

    supabase.storage.from_("scans").upload(
        path=file_name,
        file=image_bytes,
        file_options={"content-type": "image/jpeg"}
    )

    # Get the public URL (or signed URL) of the uploaded image
    public_url = supabase.storage.from_("scans").get_public_url(file_name)

    # Create a signed URL (useful when bucket becomes private)
    signed = supabase.storage.from_("scans").create_signed_url(file_name, 60 * 60 * 24)  # valid 24h
    signed_url = signed["signedURL"] if "signedURL" in signed else None

    # Save the record to the table
    supabase.table("scans").insert({
        "user_id": user_id,
        "file_name": file_name,
        "image_url": public_url,  # fallback for dev
        "signed_url": signed_url,  # secure for prod
        "prediction": predicted_label,
        "confidence": confidence
    }).execute()

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "file_name": file_name
    }

