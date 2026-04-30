# Technical Implementation Guide

## How Qwen3-TTS Speaker Encoding Works (from source code)

### 1. The Original Speaker Encoder (ECAPA-TDNN)

Located in `qwen_tts/core/models/modeling_qwen3_tts.py`, the built-in encoder uses:

- **Res2NetBlock**: Multi-scale feature extraction with split channels
  - `in_channels // scale` sub-bands processed in parallel
  - Each sub-band adds residual from previous, creating multi-resolution features
- **SqueezeExcitation (SE) blocks**: Channel-wise attention for feature recalibration
- **TDNN layers**: 1D convolutions with dilation for temporal context
- **AttentiveStatisticsPooling**: Weighted mean + std pooling over time dimension
- **Final linear**: Projects to 1024-dim, L2-normalized

Config from `configuration_qwen3_tts.py`:
```
mel_dim=128, enc_dim=1024
enc_channels=[512, 512, 512, 512, 1536]
enc_kernel_sizes=[5, 3, 3, 3, 1]
enc_dilations=[1, 2, 3, 4, 1]
enc_attention_channels=128, enc_res2net_scale=8, enc_se_channels=128
sample_rate=24000
```

### 2. How Speaker Embedding Is Used in Training (sft_12hz.py)

```python
# Step 1: Compute speaker embedding from reference mel spectrogram
speaker_embedding = model.speaker_encoder(ref_mels.to(device).to(dtype)).detach()

# Step 2: Build dual-track input embeddings
input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

# Step 3: Inject speaker embedding at position 6 in codec track
input_codec_embedding[:, 6, :] = speaker_embedding

# Step 4: Sum text and codec tracks
input_embeddings = input_text_embedding + input_codec_embedding

# Step 5: Add residual codebook embeddings (layers 1-15)
for i in range(1, 16):
    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i-1](codec_ids[:,:,i])
    input_embeddings = input_embeddings + codec_i_embedding * codec_mask.unsqueeze(-1)

# Step 6: Forward through talker LM
outputs = model.talker(inputs_embeds=input_embeddings[:,:-1,:], ...)

# Step 7: Loss = talker_loss + 0.3 * sub_talker_loss
loss = outputs.loss + 0.3 * sub_talker_loss
```

### 3. How Mel Spectrograms Are Extracted (dataset.py)

```python
# Audio must be 24kHz
mels = mel_spectrogram(
    torch.from_numpy(audio).unsqueeze(0),
    n_fft=1024, num_mels=128, sampling_rate=24000,
    hop_size=256, win_size=1024, fmin=0, fmax=12000
).transpose(1, 2)  # Shape: (1, T, 128)
```

### 4. Data Format for Fine-tuning

Input JSONL (one line per sample):
```json
{"audio": "./data/utt0001.wav", "text": "Transcript here.", "ref_audio": "./data/ref.wav"}
```

After `prepare_data.py` (adds audio_codes via tokenizer):
```json
{"audio": "...", "text": "...", "ref_audio": "...", "audio_codes": [[...], ...]}
```

---

## Implementation Plan for Custom Encoders

### What Needs to Happen

1. **Extract ground-truth embeddings**: Run Qwen3's speaker_encoder on all reference audio in the dataset. Save (mel_spectrogram, ground_truth_embedding) pairs.

2. **Train custom encoders**: For each architecture, train to minimize distance to ground-truth embeddings.

3. **Swap encoder**: Replace the speaker_encoder call in sft_12hz.py with the custom encoder. Fine-tune end-to-end.

4. **Evaluate**: Generate audio with each encoder variant and compare.

### Proposed Encoder Architectures

#### A. Linear Baseline (current placeholder — needs fixing)
```python
class LinearEncoder(nn.Module):
    def __init__(self, mel_dim=128, output_dim=1024):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over time
        self.fc = nn.Linear(mel_dim, output_dim)
    def forward(self, x):  # x: (B, T, 128)
        pooled = self.pool(x.transpose(1,2)).squeeze(-1)  # (B, 128)
        return F.normalize(self.fc(pooled), p=2, dim=-1)
```

#### B. CNN Encoder
```python
class CNNEncoder(nn.Module):
    def __init__(self, mel_dim=128, output_dim=1024):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(mel_dim, 256, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(512),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_dim)
    def forward(self, x):  # x: (B, T, 128)
        h = self.conv_layers(x.transpose(1,2))  # (B, 512, T)
        pooled = self.pool(h).squeeze(-1)  # (B, 512)
        return F.normalize(self.fc(pooled), p=2, dim=-1)
```

#### C. Transformer Encoder
```python
class TransformerEncoder(nn.Module):
    def __init__(self, mel_dim=128, output_dim=1024, nhead=8, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(mel_dim, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(512, output_dim)
    def forward(self, x):  # x: (B, T, 128)
        h = self.proj(x)
        h = self.transformer(h)
        pooled = h.mean(dim=1)  # Mean pool over time
        return F.normalize(self.fc(pooled), p=2, dim=-1)
```

#### D. Lightweight ECAPA-TDNN (reduced version of Qwen3's encoder)
- Fewer channels (256 instead of 512)
- Fewer layers
- Same architectural principles but much smaller parameter count

### Stage 1 Training Script Outline

```python
# 1. Load Qwen3-TTS model (for its speaker_encoder)
qwen3 = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", ...)

# 2. Extract ground-truth embeddings
for audio in dataset:
    mel = extract_mel(audio)  # (1, T, 128)
    with torch.no_grad():
        gt_embedding = qwen3.model.speaker_encoder(mel)  # (1, 1024)
    save(mel, gt_embedding)

# 3. Train custom encoder
custom_encoder = CNNEncoder(mel_dim=128, output_dim=1024)
optimizer = AdamW(custom_encoder.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for mel, gt_emb in dataloader:
        pred_emb = custom_encoder(mel)
        loss = 1 - F.cosine_similarity(pred_emb, gt_emb).mean()  # or MSE
        loss.backward()
        optimizer.step()
```

---

## Running the Project

### Docker Setup
```bash
# Build
docker build -t vader_env .

# Run (interactive, with GPU)
sudo docker run -d -it -v "$(pwd):/workspace" --gpus all vader_env

# Attach
docker exec -it <container_id> bash
```

### Quick Test (verify installation)
Run the first cell in `test.ipynb` — it loads `Qwen3-TTS-12Hz-1.7B-VoiceDesign` and generates a test audio file.

### Fine-tuning (existing pipeline)
```bash
# 1. Prepare data
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# 2. Train
python finetuning/sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 --lr 2e-5 --num_epochs 3
```

### HuggingFace Models to Download
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` — base model with voice cloning
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` — for baseline testing
- `Qwen/Qwen3-TTS-Tokenizer-12Hz` — audio tokenizer
