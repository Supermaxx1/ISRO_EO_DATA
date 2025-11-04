from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/siglip-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model(**inputs).last_hidden_state.mean(dim=1)
    return embeds
