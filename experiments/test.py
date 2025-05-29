# Run with: python -m experiments.test
# Tests the model with a sample image
import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import requests

# Import your model
from logs.resnet_base_old import *

# ----------------------------------------------------------
# MODEL SETUP
# ----------------------------------------------------------

# Load a pretrained ResNet-50 model with weights from ImageNet
# num_classes=1000 is implicit (ImageNet has 1000 categories)
model = resnet50_base(pretrained=True)

# Optionally, inject adapters (commented out unless you're testing them)
# add_adapters_to_resnet(model, reduction=4)

# Freeze all weights except the adapters and classification head
freeze_encoder(model)

# Put the model in evaluation mode (disables dropout, batch norm updates, etc.)
model.eval()

# ----------------------------------------------------------
# IMAGE LOADING + PREPROCESSING
# ----------------------------------------------------------

# Load a local image file and convert it to RGB (ensures 3 channels)
image = Image.open("snail-shell.jpg").convert("RGB")

# Define the preprocessing steps to match what the ResNet expects
transform = transforms.Compose([
    transforms.Resize(256),             # Resize image to 256x256 pixels
    transforms.CenterCrop(224),         # Crop the center 224x224 region
    transforms.ToTensor(),              # Convert image to a PyTorch tensor [0,1]
    transforms.Normalize(               # Normalize using ImageNet mean/std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Apply the transform to the image and add a batch dimension (1 image in batch)
input_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

# ----------------------------------------------------------
# INFERENCE
# ----------------------------------------------------------

# Disable gradient computation (saves memory and improves speed during inference)
with torch.no_grad():
    output = model(input_tensor)  # Forward pass through the model
    probs = torch.nn.functional.softmax(output[0], dim=0)  # Convert logits to probabilities

# ----------------------------------------------------------
# OUTPUT PREDICTIONS
# ----------------------------------------------------------

# Get the human-readable category labels from torchvision's ImageNet metadata
labels = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

# Get the indices of the top 5 highest probabilities
top5 = torch.topk(probs, 5)

# Print top-5 predictions and their probabilities
for idx in top5.indices:
    print(f"{labels[idx]}: {probs[idx]:.4f}")