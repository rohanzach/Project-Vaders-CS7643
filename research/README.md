# Research References

## Papers

All referenced papers from the project proposal. PDFs are stored in `papers/`.

| # | Title | arXiv | Key Contribution |
|---|-------|-------|-----------------|
| 1 | Neural Voice Cloning with a Few Samples | [1802.06006](https://arxiv.org/abs/1802.06006) | Speaker encoding vs. fine-tuning approaches for few-shot voice cloning |
| 2 | Qwen3-TTS Technical Report | [2601.15621](https://arxiv.org/abs/2601.15621) | Base TTS architecture used in this project |
| 3 | VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot TTS | [2406.05370](https://arxiv.org/abs/2406.05370) | Codec-based approach achieving human parity in zero-shot TTS |
| 4 | Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis | [2312.05882](https://arxiv.org/abs/2312.05882) | Improved prompting for zero-shot speaker adaptation |
| 5 | Meta-Voice: Fast Few-Shot Style Transfer for Expressive Voice Cloning | [2111.07218](https://arxiv.org/abs/2111.07218) | MAML-based meta-learning for few-shot voice style transfer |

## Datasets

| Dataset | URL | Description |
|---------|-----|-------------|
| LibriSpeech | https://www.openslr.org/12 | 1000 hours of read English speech from public domain audiobooks |
| VCTK Corpus | https://datashare.ed.ac.uk/handle/10283/3443 | 110 English speakers with various accents, ~400 sentences each |

## Evaluation Metrics

- **Word Error Rate (WER)**: Measures intelligibility of synthesized speech
- **Cosine Similarity**: Compares speaker embeddings between reference and generated audio
- **Human Evaluation**: Subjective ratings of speaker similarity and naturalness
