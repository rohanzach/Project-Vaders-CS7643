import torch
import torch.nn as nn
import torch.nn.functional as F

# Song
class Prenet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvGLUBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B*N, T, D)
        x = x.transpose(1, 2)  # → (B*N, D, T)

        out = self.conv(x)
        A, B = out.chunk(2, dim=1)
        out = A * torch.sigmoid(B)  # GLU

        out = out.transpose(1, 2)  # back to (B*N, T, D)

        return x.transpose(1, 2) + 0.5 * out  # residual

class TemporalPooling(nn.Module):
    def forward(self, x):
        # x: (B, N, T, D)
        return x.mean(dim=2)  # global mean pooling over T

class SampleAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        # x: (B, N, D)
        out, _ = self.attn(x, x, x)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, N, D)
        scores = self.fc(x).squeeze(-1)  # (B, N)
        scores = F.softsign(scores)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, N, 1)

        embedding = (x * weights).sum(dim=1)  # (B, D)
        return embedding

class ConvEncoder(nn.Module):
    def __init__(
        self,
        n_mels=128,
        prenet_dim=128,
        encoder_dim=2048, # Change if needed
        n_prenet=2,
        n_conv=3,
        n_heads=4
    ):
        super().__init__()

        self.prenet = Prenet(n_mels, prenet_dim, n_prenet)

        self.conv_blocks = nn.ModuleList([
            ConvGLUBlock(prenet_dim) for _ in range(n_conv)
        ])

        self.pool = TemporalPooling()

        self.sample_attn = SampleAttention(prenet_dim, n_heads)

        self.fc = nn.Linear(prenet_dim, encoder_dim)

        self.attn_pool = AttentionPooling(encoder_dim)

        self.norm = nn.LayerNorm(encoder_dim)

    def forward(self, x):
        # x: (B, T, F)

        if x.dim() == 3:
            x = x.unsqueeze(1)  # → (B, 1, T, F)
            
        B, N, T, F = x.shape

        x = x.view(B * N, T, F)

        # Prenet
        x = self.prenet(x)

        # Conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # reshape back
        x = x.view(B, N, T, -1)

        # Temporal pooling
        x = self.pool(x)  # (B, N, D)

        # Sample attention
        x = self.sample_attn(x)

        # FC
        x = self.fc(x)

        # Attention pooling across samples
        x = self.attn_pool(x)

        # Normalize (final speaker embedding)
        x = self.norm(x)

        return x  # (B, D)