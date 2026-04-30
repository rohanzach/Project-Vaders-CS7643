# Project Vaders — CS 7643 Deep Learning (Spring 2026)

## Team
| Member | Role/Focus |
|--------|-----------|
| Rohan Zacharia | Project lead, architecture proposals, two-stage training idea |
| Samuel Adegbosin (Sam) | Research, training strategy brainstorming, TA outreach for proposal feedback |
| Aditya Kommi | Cloud infrastructure setup (GPU credits), Ed/course logistics |
| Shangshang Song | Docker environment, local setup, compute |

## Project Title
**Voice Cloning with Learned Speaker Embeddings on Qwen3-TTS**

## One-Line Summary
Build a lightweight speaker encoder that learns speaker embeddings from minimal reference audio, then conditions the pre-trained Qwen3-TTS model to generate cloned speech — comparing multiple encoder architectures.

---

## Problem Statement
Voice cloning from only a few seconds of reference audio remains challenging, especially in preserving speaker identity across diverse utterances. Current state-of-the-art models like Qwen3-TTS use large, complex speaker encoders (ECAPA-TDNN with Res2Net blocks, ~1024-dim embeddings from 128-dim mel spectrograms). The project investigates whether smaller, alternative encoder architectures can produce embeddings close enough to the original to maintain voice cloning quality.

## Approach (Two-Stage Training — Rohan's Proposal)

### Stage 1: Speaker Embedding Distillation
- Train a custom (smaller) speaker encoder to **mimic** the Qwen3-TTS built-in speaker encoder (ECAPA-TDNN based, Res2NetBlock architecture)
- Ground truth = the embedding vector produced by Qwen3-TTS's own `speaker_encoder` (from `ref_mels`)
- Loss: MSE or cosine distance between custom encoder output and original encoder output
- Input: mel spectrograms (128-dim) from reference audio
- Output: 1024-dim normalized speaker embedding vector

### Stage 2: End-to-End Fine-tuning
- Plug the custom encoder into the full Qwen3-TTS pipeline
- Fine-tune end-to-end with audio generation
- Evaluate generated audio quality using metrics below
- Compare across different encoder architectures (CNN, Transformer, simple linear, etc.)

## Architecture Details

### Qwen3-TTS Pipeline (from the paper & code)
1. **Text processing**: Standard Qwen tokenizer
2. **Audio tokenization**: Qwen-TTS-Tokenizer-12Hz (12.5 Hz, 16-layer multi-codebook, semantic + acoustic)
3. **Speaker encoder**: ECAPA-TDNN based (Res2NetBlock, SqueezeExcitation, AttentiveStatisticsPooling)
   - Input: 128-dim mel spectrogram
   - Output: 1024-dim speaker embedding (L2-normalized)
   - Config: `enc_channels=[512, 512, 512, 512, 1536]`, `enc_kernel_sizes=[5, 3, 3, 3, 1]`, `enc_dilations=[1, 2, 3, 4, 1]`
4. **Dual-track LM**: Qwen3 backbone with text + codec token tracks
5. **MTP Module**: Multi-Token Prediction for residual codebook layers
6. **Streaming Codec Decoder**: Converts tokens back to waveforms

### Current Custom Encoder (VaderSpeakerEncoder — Placeholder)
```python
class VaderSpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=1024):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        embedding = self.projection(x)
        return torch.nn.functional.normalize(embedding, p=2, dim=-1)
```
**NOTE**: This is a bare-minimum placeholder. The real Qwen3 encoder uses 128-dim mel input, not 80. This needs to be updated and expanded with real architectures.

### How the Speaker Embedding Integrates (from sft_12hz.py)
```python
# The speaker embedding is computed from reference mel spectrograms
speaker_embedding = model.speaker_encoder(ref_mels)
# It's injected at position 6 in the codec embedding sequence
input_codec_embedding[:, 6, :] = speaker_embedding
# Then combined with text embeddings for the dual-track input
input_embeddings = input_text_embedding + input_codec_embedding
```

## Datasets
| Dataset | Size | Description |
|---------|------|-------------|
| **LibriSpeech** | 1000 hours | Read English speech from public domain audiobooks |
| **VCTK Corpus** | 110 speakers × ~400 sentences | English speakers with various accents |

## Evaluation Metrics
- **Word Error Rate (WER)**: Intelligibility of synthesized speech
- **Cosine Similarity**: Between reference and generated speaker embeddings
- **Human Evaluation**: Subjective ratings of speaker similarity and naturalness

## Key References
1. "Neural Voice Cloning with a Few Samples" (arXiv:1802.06006) — speaker encoding vs. fine-tuning
2. "Qwen3-TTS Technical Report" (arXiv:2601.15621) — base architecture
3. "VALL-E 2" (arXiv:2406.05370) — codec-based zero-shot TTS
4. "Boosting Prompting for Zero-Shot Speech Synthesis" (arXiv:2312.05882)
5. "Meta-Voice" (arXiv:2111.07218) — MAML-based meta-learning for voice style transfer

---

## Repository Structure
```
Project-Vaders-CS7643/
├── Dockerfile                  # CUDA 12.4, Python 3.10, Flash-Attention build
├── pyproject.toml              # Package: qwen-tts, deps: transformers, accelerate, librosa, torchaudio
├── test.ipynb                  # Baseline test: English voice design generation
├── Group Proposal.docx         # Submitted project proposal
│
├── qwen_tts/                   # Main package (forked from Qwen3-TTS)
│   ├── core/
│   │   ├── models/
│   │   │   ├── configuration_qwen3_tts.py    # Config classes (SpeakerEncoder, Talker, CodePredictor)
│   │   │   ├── modeling_qwen3_tts.py         # Full model: Res2NetBlock, ECAPA-TDNN, Talker, etc.
│   │   │   ├── processing_qwen3_tts.py       # Processor
│   │   │   └── speaker_embedding.py          # ** OUR CUSTOM ENCODER ** (VaderSpeakerEncoder — placeholder)
│   │   ├── tokenizer_12hz/                   # 12.5Hz multi-codebook tokenizer
│   │   └── tokenizer_25hz/                   # 25Hz single-codebook tokenizer (VQ, Whisper encoder)
│   └── inference/
│       ├── qwen3_tts_model.py                # High-level model wrapper (from_pretrained, generate_*)
│       └── qwen3_tts_tokenizer.py            # Tokenizer wrapper
│
├── finetuning/
│   ├── README.md               # Fine-tuning instructions
│   ├── prepare_data.py         # JSONL → audio_codes extraction
│   ├── dataset.py              # TTSDataset with mel extraction, collate_fn
│   └── sft_12hz.py             # SFT training script (AdamW, gradient accumulation, checkpointing)
│
├── examples/
│   ├── test_model_12hz_base.py
│   ├── test_model_12hz_custom_voice.py
│   ├── test_model_12hz_voice_design.py
│   └── test_tokenizer_12hz.py
│
├── research/
│   ├── README.md               # Paper references, datasets, metrics table
│   └── papers/                 # PDF copies of referenced papers
│
└── assets/
    └── Qwen3_TTS.pdf           # Qwen3-TTS technical report
```

## Infrastructure

### Docker (Local)
- **Image**: `nvidia/cuda:12.4.1-devel-ubuntu22.04` base, Flash-Attention 2, PyTorch with CUDA 12.4
- **Run**: `sudo docker run -d -it -v "$(pwd):/workspace" --gpus all vader_env`

### PACE-ICE (Georgia Tech GPU Cluster) — Ed Post #240
- **Access**: Connect to GaTech VPN (GlobalProtect) → `ssh <gtid>@login-ice.pace.gatech.edu`
- **GPUs available**: V100 (32GB, easier to get), A100 (faster but more contested)
- **Resource limits**: Max 512 CPU hours, 16 GPU hours per job, 8-hour max walltime
- **Storage**: Home = 15GB only; **use ~/scratch** (300GB) for models, venvs, data
- **Workflow**: ssh → `salloc -N1 -t0:15:00 --gres=gpu:V100:1 --mem-per-gpu=32G` → `module load anaconda3` → activate env → run
- **VSCode SSH**: Can connect via Remote-SSH extension for IDE workflow
- **Batch jobs**: `sbatch myscript.sh` for longer training runs
- **Key tip**: Always install packages and download models to ~/scratch to avoid quota issues

### Google Cloud Credits — Ed Post #779
- **GCP coupon** available for students: [Student Coupon Retrieval Link](https://vector.my.salesforce-sites.com/GCPEDU?cid=M02zAw3yobo3ic4AWddQxUrEWdiP0w%2FZbIFB45dJs2fNqOTsCO5ousxEOyFbbrQa/)
- Use your @gatech.edu email to request
- One coupon per person
- **Important**: Set spending limit to $0 if entering credit card to avoid surprise charges
- **Note**: Need to request GPU quota increase in GCP console
- Save credits for the project (not needed for assignments)

### Google Colab
- Free tier available, can also use with GCP credits for better GPUs

### HuggingFace Models
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (voice cloning capable)
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (fine-tuned for custom voices)
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (instruction-following voice generation)
- `Qwen/Qwen3-TTS-Tokenizer-12Hz` (audio tokenizer)

---

## Final Report Requirements (60 points)
The report must be 4-6 pages, conference-style (CVPR/ECCV format), using the provided LaTeX template.

| Section | Points | What to Address |
|---------|--------|-----------------|
| Introduction/Background | 5 | What problem, no jargon |
| Current Practice & Limits | 5 | How voice cloning is done today |
| Motivation / Impact | 5 | Why this matters |
| Data Description | 5 | Dataset details per Datasheets for Datasets |
| Approach | 10 | Exact method, why it should work, what's novel |
| Challenges | 5 | Anticipated and encountered problems |
| Experiments & Results | 10 | Metrics, quantitative + qualitative, success/failure analysis |
| Figures/Tables | 5 | Clear visualizations |
| Clarity | 5 | Self-contained, peer-understandable |
| DL Understanding | 5 | Architecture rationale, loss, overfitting, hyperparams, framework |

Additional requirements: Team contributions table, comprehensive references section (neither counts toward page limit).

**LaTeX Template**: https://www.overleaf.com/read/fdjpfsdhztfp
**Deadline**: ~May 2, 2026 (submit PDF on Gradescope as group assignment)

---

## Key EdStem Insights

### Dataset Size (Ed Post #292)
- **Minimum ~50,000 samples** recommended (using assignments as a guideline)
- More complex models need more samples
- LibriSpeech (1000 hrs) and VCTK (110 speakers × 400 sentences = ~44,000 utterances) should be sufficient

### Team of 4 Expectations (Ed Post #633)
- Bigger teams expected to take on **more ambitious** projects
- Each member must contribute original code roughly equal to **one programming assignment**
- Everyone must contribute technically (not just report writing)

### Proposal Feedback (Ed pinned post)
- **Project proposal feedback released on Gradescope** — check there!
- Teams should begin working without waiting for feedback (don't let it block progress)

### Google Cloud Credits (Ed Post #779)
- Student coupon available via GaTech email
- Save for project, not assignments

### PACE-ICE (Ed Post #240)
- Full guide for GPU cluster access — see Infrastructure section above
