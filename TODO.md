# Project Vaders — Task Tracker

## Phase 1: Research & Planning
- [x] Validate two-stage training approach feasibility — confirmed viable
- [ ] Research audio evaluation metrics (WER, cosine similarity, MOS, PESQ, etc.)
- [x] Decide on encoder architectures to compare — Qwen3 baseline + 3 custom (one from paper + 2 more from Sam/Aditya)
- [ ] Set up cloud GPU instance using course credits (Aditya)
- [ ] Get proposal feedback from TA/CA (Sam reaching out)
- [x] Finalize training data strategy — LibriSpeech train-clean-100 (100 hours), dev-clean for testing

## Phase 2: Implementation (Current)
- [ ] Fix `VaderSpeakerEncoder` — update input_dim from 80 to 128 (mel_dim), implement real architectures
- [x] Write ground-truth extraction pipeline (Rohan — Colab notebook, extracts embeddings via Qwen3 speaker_encoder, saves as .safetensors)
- [x] Prepare training data: extract mel spectrograms + ground-truth speaker embeddings from Qwen3's encoder (Rohan — done for dev-clean, train-clean-100 next)
- [ ] Write Stage 1 training loop (Shangshang — in progress, troubleshooting)
- [ ] Implement additional encoder architectures (Sam — 1, Aditya — 1)
- [ ] Integrate training code with Rohan's Colab structure (Shangshang)
- [ ] Write Stage 2: swap custom encoder into full Qwen3 pipeline for audio generation (Rohan — starting)
- [ ] Set up evaluation pipeline (WER via ASR model, cosine similarity, human eval)

## Phase 3: Experiments
- [ ] Train each encoder architecture variant
- [ ] Record cosine similarity between custom embeddings and ground truth
- [ ] Generate audio samples with each encoder and evaluate
- [ ] Compare against baseline (Qwen3's built-in encoder)
- [ ] Ablation studies (embedding dimension, training data size, etc.)

## Phase 4: Report & Submission
- [ ] Set up LaTeX report using Overleaf template (https://www.overleaf.com/read/fdjpfsdhztfp)
- [ ] Write Introduction / Background / Motivation (20 points)
- [ ] Write Approach section (15 points)
- [ ] Write Experiments & Results (10 points)
- [ ] Create figures and tables (architecture diagrams, training curves, comparison tables)
- [ ] Write team contributions table
- [ ] Compile references
- [ ] Submit PDF to Gradescope "Final Project" (group assignment, add all members)
- [ ] Submit supplementary code zip to Gradescope

## Key Dates
- **Sunday Meeting**: 2:00 PM — working session, combine code, run first epoch
- **Recurring**: Tuesday + Thursday meetings, weekend sessions
- **Deadline**: ~May 2, 2026
