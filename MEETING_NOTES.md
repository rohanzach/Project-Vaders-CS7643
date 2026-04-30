# Meeting Notes — Project Vaders

## Weekday Meeting (Pre-Sunday Planning Session)

**Attendees**: Rohan Zacharia, Samuel Adegbosin, Aditya Kommi, Shangshang Song (joined late)
**Duration**: ~28 minutes

### Key Discussion Points

1. **Two-Stage Training Approach (Rohan)**
   - Stage 1: Train a custom model to replicate the Qwen3 speaker embedding. Ground truth = the embedding vector from Qwen3's built-in speaker encoder. Goal: can our smaller model produce the same vector from the same audio input?
   - Stage 2: End-to-end training with actual audio generation. Use something VAE-like where the model tries to recreate the audio, then compare. Need to research appropriate audio comparison metrics.

2. **Architecture Exploration**
   - The project can showcase different encoder architectures (convolution, transformer, etc.) and compare which produces the best embedding approximation.
   - Results don't need to be perfect — showing the comparison across architectures is valuable.

3. **Training Compute Concerns**
   - Audio generation is slow (even a single sample takes noticeable time on a laptop).
   - Training on 1000+ audio samples locally would take forever.
   - Aditya mentioned cloud GPU credits available through the course (Ed lessons).

4. **Data Strategy**
   - Use the same datasets the papers used (LibriSpeech, VCTK).
   - Need to ensure the work is differentiated enough from existing papers — can't just replicate what's been done.

5. **Proposal Feedback**
   - No feedback received from TAs yet (submitted over a week ago).
   - Sam found an Ed post saying teams should begin working without waiting for feedback.
   - Sam will still reach out to a TA/CA for feedback.
   - Aditya concerned about direction — wants confirmation they're on the right track.

6. **Shangshang's Computer Issues**
   - Computer with Docker setup broke; planning DIY fix or new computer purchase.
   - Cloud credits would help bypass local machine dependency.

### Action Items

| Owner | Task | Status |
|-------|------|--------|
| **Aditya** | Research cloud GPU credits from Ed lessons, set up shared cloud instance | Assigned |
| **Sam** | Reach out to TA/CA for proposal feedback | Assigned |
| **Sam** | Brainstorm training strategies and research what to do differently | Assigned |
| **Rohan** | Research training approach feasibility, validate two-stage idea | Assigned |
| **Shangshang** | Fix/replace computer, set up Docker locally when ready | Assigned |
| **All** | Research: how to train the model, what metrics to use for audio evaluation, which datasets | Assigned |

### Next Meeting
- **When**: Sunday at 2:00 PM
- **Goal**: Settle on a concrete strategy and start dividing implementation work
- **Project Deadline**: ~May 2, 2026 (~3 weeks from this meeting)

---

## Mid-Week Progress Meeting (~32 min)

**Attendees**: Rohan Zacharia, Samuel Adegbosin, Aditya Kommi, Shangshang Song
**Format**: Working session — code demos and screen sharing

### Key Progress

1. **Rohan's Google Colab Pipeline (Demo)**
   - Built a complete Colab notebook for Stage 1 ground-truth extraction
   - **Setup**: Connects to Google Drive, clones the GitHub repo, caches Qwen3-TTS model download (~3GB+) to avoid re-downloading each session
   - **`create_speaker_embedding` function**: Traverses LibriSpeech directory structure (speaker → book → audio files), runs each audio file through Qwen3's pre-trained speaker encoder, collects all embeddings
   - **Output**: Saves all speaker embeddings as a `.safetensors` file — this is the ground-truth training data for Stage 1
   - **Dataset**: Downloaded LibriSpeech `dev-clean` for testing; plans to use `train-clean-100` (100 hours) for real training
   - Code has been pushed to GitHub

2. **Shangshang's Training Loop (Demo)**
   - Working on the custom encoder training loop (Stage 1)
   - Built a pre-processing step that loads audio files from a folder and converts them to mel spectrograms
   - Architecture: Following a paper's structure for converting mel spectrograms → speaker embeddings
   - The training loop will use Rohan's extracted ground-truth embeddings as targets
   - Still troubleshooting — will ping team on Teams when ready

3. **Architecture Discussion**
   - Team agreed on plan: Qwen3's original encoder as baseline + 3 custom architectures (total of 4 to compare)
   - Shangshang has one architecture from the paper
   - Sam and Aditya will each create additional architecture variants
   - Plan: "plug and play" — swap different encoder architectures into the same training pipeline

4. **Data Size Confirmation**
   - Aditya mentioned the Ed post requiring ~50,000 samples
   - LibriSpeech train-clean-100 (100 hours) should be more than enough
   - Larger subsets (360h, 500h) available if needed

5. **Stage 2 Planning (Rohan)**
   - Rohan will start working on the embedding swap — plugging the custom encoder into the full Qwen3-TTS pipeline for end-to-end audio generation
   - Knows where in the code to swap the speaker encoder; needs to test it
   - This is the "final step" but wants to get a head start

6. **Collaboration Setup**
   - Rohan and Shangshang shared their Google Colabs with the team via Gmail
   - Plan to consolidate code into the GitHub repo
   - Shangshang will also clone the repo and try integrating her code with Rohan's structure

### Action Items

| Owner | Task | Status |
|-------|------|--------|
| **Rohan** | Ground-truth embedding extraction pipeline (Colab) | ✅ Done |
| **Rohan** | Start working on Stage 2 — embedding swap in full Qwen3 pipeline | In Progress |
| **Shangshang** | Finish training loop, troubleshoot, ping team on Teams when ready | In Progress |
| **Shangshang** | Clone repo and integrate training code with Rohan's Colab structure | Assigned |
| **Sam** | Create an alternative encoder architecture variant | Assigned |
| **Aditya** | Create an alternative encoder architecture variant | Assigned |
| **Aditya** | Review shared Colabs before Sunday meeting | Assigned |
| **All** | Share Google Colabs with each other (via Gmail) | ✅ Done |

### Schedule
- **Next Meeting**: Sunday at 2:00 PM — working session to combine code and run first epoch
- **Goal for this weekend**: Have a working training pipeline that can run at least one epoch
- **Going forward**: Tuesday + Thursday meetings, plus weekend sessions
- **Project Deadline**: ~May 2, 2026
