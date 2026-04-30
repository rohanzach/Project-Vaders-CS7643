import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class TDNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, dim, kernel_size=1)

    def forward(self, x):
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ConvGLUBlock(nn.Module):
    def __init__(self, dim, dilation=1, scale=8):
        super().__init__()
        if dim % scale != 0:
            raise ValueError("dim must be divisible by scale for ECAPA-style split processing")

        self.width = dim // scale
        self.scale = scale
        self.pre = TDNNBlock(dim, dim, kernel_size=1)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=3, padding=dilation, dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        self.post = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
        )
        self.se = SqueezeExcitation(dim)

    def forward(self, x):
        # x: (B*N, T, D)
        residual = x
        x = x.transpose(1, 2)  # -> (B*N, D, T)
        x = self.pre(x)

        splits = torch.split(x, self.width, dim=1)
        outputs = []
        running = None

        for i, split in enumerate(splits):
            if i == 0:
                outputs.append(split)
                running = split
            else:
                running = split if i == 1 else running + split
                running = F.relu(self.bns[i - 1](self.convs[i - 1](running)))
                outputs.append(running)

        out = torch.cat(outputs, dim=1)
        out = self.post(out)
        out = self.se(out)
        out = out.transpose(1, 2)  # -> (B*N, T, D)

        return residual + out
    
class TemporalPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, T, D)
        scores = torch.tanh(self.linear1(x))
        scores = self.linear2(scores)  # (B, N, T, D)
        weights = F.softmax(scores, dim=2)

        mean = (x * weights).sum(dim=2)
        var = ((x - mean.unsqueeze(2)) ** 2 * weights).sum(dim=2).clamp(min=1e-5)
        std = torch.sqrt(var)

        return torch.cat([mean, std], dim=-1)  # attentive statistics pooling

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
    
class TDNNSpeakerEncoder(nn.Module):
    def __init__(
        self,
        n_mels=128,
        prenet_dim=128,
        encoder_dim=2048,
        n_prenet=2,
        n_conv=3,
        n_heads=4
    ):
        super().__init__()

        self.n_mels = n_mels
        self.prenet = Prenet(n_mels, prenet_dim, n_prenet)

        self.conv_blocks = nn.ModuleList([
            ConvGLUBlock(prenet_dim, dilation=i + 2) for i in range(n_conv)
        ])

        self.mfa_dim = prenet_dim * 3
        self.mfa = nn.Sequential(
            nn.Conv1d(prenet_dim * n_conv, self.mfa_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.mfa_dim),
        )

        self.pool = TemporalPooling(self.mfa_dim)

        self.sample_attn = SampleAttention(self.mfa_dim * 2, n_heads)

        self.fc = nn.Linear(self.mfa_dim * 2, encoder_dim)

        self.attn_pool = AttentionPooling(encoder_dim)

        self.norm = nn.LayerNorm(encoder_dim)


    def forward(self, x):
        # Accept either a single utterance per sample (B, T, F)
        # or multiple utterances grouped per sample (B, N, T, F).
        if x.dim() == 3:
            # Handle both (B, T, F) and (B, F, T) layouts.
            if x.shape[-1] == self.n_mels:
                x = x.unsqueeze(1)
            elif x.shape[1] == self.n_mels:
                x = x.transpose(1, 2).unsqueeze(1)
            else:
                raise ValueError(
                    f"Expected 3D input shaped (B, T, {self.n_mels}) or (B, {self.n_mels}, T), got {tuple(x.shape)}"
                )
        elif x.dim() != 4:
            raise ValueError(
                f"TDNNSpeakerEncoder expected input with shape (B, T, F) or (B, N, T, F), got {tuple(x.shape)}"
            )

        # Handle both (B, N, T, F) and (B, N, F, T) layouts.
        if x.shape[-1] != self.n_mels:
            if x.shape[2] == self.n_mels:
                x = x.transpose(2, 3)
            else:
                raise ValueError(
                    f"Expected mel dimension {self.n_mels} on last axis, got shape {tuple(x.shape)}"
                )

        B, N, T, mel_dim = x.shape

        x = x.reshape(B * N, T, mel_dim)

        # Prenet
        x = self.prenet(x)

        # ECAPA-style temporal blocks with multi-layer feature aggregation
        block_outputs = []
        for block in self.conv_blocks:
            x = block(x)
            block_outputs.append(x)

        x = torch.cat(block_outputs, dim=-1)
        x = self.mfa(x.transpose(1, 2)).transpose(1, 2)

        # reshape back
        x = x.reshape(B, N, T, -1)

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