from vision_encoder import extract_image_features
from fusion_layer import ProjectionAlignment
from text_model import get_text_embeddings
import torch

projection = ProjectionAlignment()
projection.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

image_embeds = extract_image_features("data/sample_image.tif")
aligned_embeds = projection(image_embeds)

text_embeds = get_text_embeddings("What crop is grown here?")

print("Aligned image embedding shape:", aligned_embeds.shape)
print("Text embedding shape:", text_embeds.shape)
