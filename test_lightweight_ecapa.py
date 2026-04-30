"""
Quick local test of the Lightweight ECAPA-TDNN model.
Run: python test_lightweight_ecapa.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building blocks (same as notebook cell 16) ---

class _TimeDelayNet(nn.Module):
    """1-D convolution + ReLU (TDNN layer)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              dilation=dilation, padding="same", padding_mode="reflect")
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
    """Channel-wise recalibration via global average -> bottleneck -> sigmoid."""
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
    """TDNN -> Res2Net -> TDNN -> SE, with residual connection."""
    def __init__(self, in_ch, out_ch, scale=4, se_channels=64, kernel_size=3, dilation=1):
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
        mean = x.mean(dim=2, keepdim=True).expand_as(x)
        std = x.std(dim=2, keepdim=True).clamp(min=self.eps).expand_as(x)
        attn_in = torch.cat([x, mean, std], dim=1)
        attn = F.softmax(self.conv(self.tanh(self.tdnn(attn_in))), dim=2)
        w_mean = (attn * x).sum(dim=2)
        w_std = torch.sqrt((attn * (x - w_mean.unsqueeze(2)).pow(2)).sum(dim=2).clamp(self.eps))
        return torch.cat([w_mean, w_std], dim=1).unsqueeze(2)


class LightweightECAPATDNN(nn.Module):
    """
    ~4x smaller variant of Qwen3's ECAPA-TDNN speaker encoder.
    256 channels (vs 512), 2 SE-Res2Net blocks (vs 3), scale=4 (vs 8).
    Input:  (B, T, 128) mel spectrogram
    Output: (B, 1024) L2-normalised speaker embedding
    """
    def __init__(self, mel_dim=128, enc_dim=1024, enc_channels=None,
                 enc_kernel_sizes=None, enc_dilations=None,
                 res2net_scale=4, se_channels=64, attn_channels=64):
        super().__init__()
        if enc_channels is None:
            enc_channels = [256, 256, 256, 768]
        if enc_kernel_sizes is None:
            enc_kernel_sizes = [5, 3, 3, 1]
        if enc_dilations is None:
            enc_dilations = [1, 2, 3, 1]
        assert len(enc_channels) == len(enc_kernel_sizes) == len(enc_dilations)

        self.blocks = nn.ModuleList()
        self.blocks.append(_TimeDelayNet(mel_dim, enc_channels[0],
                                         enc_kernel_sizes[0], enc_dilations[0]))
        for i in range(1, len(enc_channels) - 1):
            self.blocks.append(_SERes2NetBlock(
                enc_channels[i - 1], enc_channels[i],
                scale=res2net_scale, se_channels=se_channels,
                kernel_size=enc_kernel_sizes[i], dilation=enc_dilations[i]))

        mfa_in = sum(enc_channels[1:-1])
        self.mfa = _TimeDelayNet(mfa_in, enc_channels[-1],
                                 enc_kernel_sizes[-1], enc_dilations[-1])
        self.asp = _AttentiveStatisticsPooling(enc_channels[-1], attn_channels)
        self.fc = nn.Conv1d(enc_channels[-1] * 2, enc_dim, kernel_size=1,
                            padding="same", padding_mode="reflect")

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 128, T)
        intermediates = []
        for layer in self.blocks:
            x = layer(x)
            intermediates.append(x)
        x = torch.cat(intermediates[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.fc(x).squeeze(-1)
        return F.normalize(x, p=2, dim=-1)


# --- Run test ---
if __name__ == "__main__":
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    model = LightweightECAPATDNN(mel_dim=128, enc_dim=1024).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Lightweight ECAPA-TDNN: {n_params:,} params")
    print(f"Qwen3 original:        ~6,200,000 params")
    print(f"Reduction:             ~{6_200_000 / n_params:.1f}x smaller")

    # Forward pass with dummy data
    dummy = torch.randn(2, 300, 128, device=device)
    out = model(dummy)
    print(f"\nForward pass test:")
    print(f"  Input:  {dummy.shape}")
    print(f"  Output: {out.shape}  (norm ≈ {out.norm(dim=-1).mean():.4f})")

    # Verify output properties
    assert out.shape == (2, 1024), f"Expected (2, 1024), got {out.shape}"
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, device=device), atol=1e-5), "Output not L2-normalized"

    print("\n✓ All checks passed! Model builds and runs correctly.")
