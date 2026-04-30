import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# Aditya's contributions: VaderSpeakerEncoder + LightweightECAPATDNN
# ---------------------------------------------------------------------------

def get_device():
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Linear Baseline (original placeholder — fixed to use mel_dim=128)
# ---------------------------------------------------------------------------

class VaderSpeakerEncoder(nn.Module):
    """Bare-minimum linear projection baseline."""

    def __init__(self, input_dim=128, output_dim=1024):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (B, T, 128)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, 128)
        embedding = self.projection(pooled)                 # (B, 1024)
        return F.normalize(embedding, p=2, dim=-1)


# ---------------------------------------------------------------------------
# 2. Lightweight ECAPA-TDNN  (Aditya Kommi)
# ---------------------------------------------------------------------------
# Same architectural building blocks as Qwen3's encoder but roughly 4×
# smaller: halved channel widths, one fewer SE-Res2Net layer, scale=4
# instead of 8, and smaller SE bottleneck.
# ---------------------------------------------------------------------------

class _TimeDelayNet(nn.Module):
    """1-D convolution + ReLU (TDNN layer)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class _Res2NetBlock(nn.Module):
    """Multi-scale feature extraction via split-transform-merge."""

    def __init__(self, channels, scale=4, kernel_size=3, dilation=1):
        super().__init__()
        assert channels % scale == 0
        width = channels // scale
        self.scale = scale
        self.blocks = nn.ModuleList([
            _TimeDelayNet(width, width, kernel_size, dilation)
            for _ in range(scale - 1)
        ])

    def forward(self, x):
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                out = chunk
            elif i == 1:
                out = self.blocks[i - 1](chunk)
            else:
                out = self.blocks[i - 1](chunk + out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class _SqueezeExcitation(nn.Module):
    """Channel-wise recalibration via global average → bottleneck → sigmoid."""

    def __init__(self, channels, se_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, se_channels, 1, padding="same", padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, channels, 1, padding="same", padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = self.sigmoid(self.conv2(self.relu(self.conv1(s))))
        return x * s


class _SERes2NetBlock(nn.Module):
    """TDNN → Res2Net → TDNN → SE, with residual connection."""

    def __init__(self, in_ch, out_ch, scale=4, se_channels=64,
                 kernel_size=3, dilation=1):
        super().__init__()
        self.tdnn1 = _TimeDelayNet(in_ch, out_ch, kernel_size=1, dilation=1)
        self.res2net = _Res2NetBlock(out_ch, scale, kernel_size, dilation)
        self.tdnn2 = _TimeDelayNet(out_ch, out_ch, kernel_size=1, dilation=1)
        self.se = _SqueezeExcitation(out_ch, se_channels)

    def forward(self, x):
        return self.se(self.tdnn2(self.res2net(self.tdnn1(x)))) + x


class _AttentiveStatisticsPooling(nn.Module):
    """Attentive mean + std pooling over the time axis."""

    def __init__(self, channels, attn_channels=64):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = _TimeDelayNet(channels * 3, attn_channels, kernel_size=1, dilation=1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attn_channels, channels, 1, padding="same", padding_mode="reflect")

    def forward(self, x):
        # x: (B, C, T)
        mean = x.mean(dim=2, keepdim=True).expand_as(x)
        std = x.std(dim=2, keepdim=True).clamp(min=self.eps).expand_as(x)

        attn_in = torch.cat([x, mean, std], dim=1)          # (B, 3C, T)
        attn = F.softmax(self.conv(self.tanh(self.tdnn(attn_in))), dim=2)  # (B, C, T)

        w_mean = (attn * x).sum(dim=2)                       # (B, C)
        w_std = torch.sqrt(
            (attn * (x - w_mean.unsqueeze(2)).pow(2)).sum(dim=2).clamp(self.eps)
        )
        return torch.cat([w_mean, w_std], dim=1).unsqueeze(2)  # (B, 2C, 1)


class LightweightECAPATDNN(nn.Module):
    """
    A ~4× smaller variant of Qwen3's ECAPA-TDNN speaker encoder.

    Changes vs. original:
      - 256 channels instead of 512  (halved)
      - 2 SE-Res2Net blocks instead of 3  (one fewer layer)
      - Res2Net scale = 4 instead of 8
      - SE bottleneck = 64 instead of 128
      - MFA aggregation channel = 768 instead of 1536

    Input:  (B, T, 128) mel spectrogram   (same as Qwen3)
    Output: (B, 1024) L2-normalised speaker embedding
    """

    def __init__(
        self,
        mel_dim: int = 128,
        enc_dim: int = 1024,
        enc_channels: list[int] | None = None,
        enc_kernel_sizes: list[int] | None = None,
        enc_dilations: list[int] | None = None,
        res2net_scale: int = 4,
        se_channels: int = 64,
        attn_channels: int = 64,
    ):
        super().__init__()

        if enc_channels is None:
            enc_channels = [256, 256, 256, 768]
        if enc_kernel_sizes is None:
            enc_kernel_sizes = [5, 3, 3, 1]
        if enc_dilations is None:
            enc_dilations = [1, 2, 3, 1]

        assert len(enc_channels) == len(enc_kernel_sizes) == len(enc_dilations)

        self.blocks = nn.ModuleList()

        # Initial TDNN layer
        self.blocks.append(
            _TimeDelayNet(mel_dim, enc_channels[0],
                          enc_kernel_sizes[0], enc_dilations[0])
        )

        # SE-Res2Net layers (2 blocks instead of Qwen3's 3)
        for i in range(1, len(enc_channels) - 1):
            self.blocks.append(
                _SERes2NetBlock(
                    enc_channels[i - 1], enc_channels[i],
                    scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=enc_kernel_sizes[i],
                    dilation=enc_dilations[i],
                )
            )

        # Multi-layer feature aggregation
        mfa_in = sum(enc_channels[1:-1])  # concat intermediate layer outputs
        self.mfa = _TimeDelayNet(
            mfa_in, enc_channels[-1],
            enc_kernel_sizes[-1], enc_dilations[-1],
        )

        # Attentive Statistics Pooling
        self.asp = _AttentiveStatisticsPooling(enc_channels[-1], attn_channels)

        # Final projection → 1024-dim
        self.fc = nn.Conv1d(
            enc_channels[-1] * 2, enc_dim,
            kernel_size=1, padding="same", padding_mode="reflect",
        )

    def forward(self, x):
        # x: (B, T, 128) — mel spectrogram
        x = x.transpose(1, 2)  # → (B, 128, T)

        intermediates = []
        for layer in self.blocks:
            x = layer(x)
            intermediates.append(x)

        # Concat outputs of SE-Res2Net blocks (skip initial TDNN)
        x = torch.cat(intermediates[1:], dim=1)
        x = self.mfa(x)

        # Pool over time
        x = self.asp(x)          # (B, 2*enc_channels[-1], 1)

        # Project
        x = self.fc(x).squeeze(-1)  # (B, enc_dim)

        # L2-normalize to match Qwen3 interface
        return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Quick param-count check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    model = LightweightECAPATDNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LightweightECAPATDNN: {n_params:,} params")

    dummy = torch.randn(2, 300, 128, device=device)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  (norm ≈ {out.norm(dim=-1).mean():.4f})")
