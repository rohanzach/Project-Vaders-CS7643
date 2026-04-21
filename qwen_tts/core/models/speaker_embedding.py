import torch
import torch.nn as nn

class BasicSpeakerEncoder(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=512, output_dim=2048):
      super().__init__()
      self.projection = nn.Sequential(
          nn.Linear(input_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, output_dim)
      )

    def forward(self, x):
        
      embedding = self.projection(x) 
      embedding = embedding.mean(dim=1) # Pool the time dimension

      return torch.nn.functional.normalize(embedding, p=2, dim=-1)