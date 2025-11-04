import torch
import torch.nn as nn

class ProjectionAlignment(nn.Module):
    def __init__(self, vision_dim=768, text_dim=1024, hidden_dim=512):
        super(ProjectionAlignment, self).__init__()
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, text_dim)

    def forward(self, vision_embeds):
        x = self.fc1(vision_embeds)
        x = self.relu(x)
        x = self.fc2(x)
        return x
