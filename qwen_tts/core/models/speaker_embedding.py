import torch
import torch.nn as nn

class VaderSpeakerEncoder(nn.Module):

    def __init__(self, input_dim=80, output_dim=1024):
      super().__init__()
      self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        
      embedding = self.projection(x) 

      return torch.nn.functional.normalize(embedding, p=2, dim=-1)