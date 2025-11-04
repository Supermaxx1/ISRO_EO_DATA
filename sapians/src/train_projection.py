import torch
from torch.utils.data import Dataset, DataLoader
from fusion_layer import ProjectionAlignment
from vision_encoder import extract_image_features
from text_model import get_text_embeddings

class EODataset(Dataset):
    def __init__(self, csv_file):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        caption = self.data.iloc[idx, 1]
        image_embeds = extract_image_features(img_path).cpu()
        text_embeds = get_text_embeddings(caption).cpu()
        return image_embeds.squeeze(0), text_embeds.squeeze(0)

# Hyperparameters
batch_size = 4
lr = 2e-4
num_epochs = 10

# Initialize
dataset = EODataset("data/captions.csv")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
projection = ProjectionAlignment().float()
optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

for epoch in range(num_epochs):
    for image_embeds, text_embeds in dataloader:
        projected = projection(image_embeds)
        loss = loss_fn(projected, text_embeds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model weights
torch.save(projection.state_dict(), "models/projection.pt")
