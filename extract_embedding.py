import torch
import torchaudio
from qwen_tts import Qwen3TTSModel
import os
from safetensors.torch import save_file, load_file
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from functools import lru_cache



@lru_cache(maxsize=None)
def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
    mel_fb = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return torch.from_numpy(mel_fb).float()

@lru_cache(maxsize=None)
def get_hann_window(win_size):
    return torch.hann_window(win_size)


def extract_mel(audio_path, sr=24000, n_fft=1024, n_mels=128,
                hop_size=256, win_size=1024, fmin=0, fmax=12000):
    wav, orig_sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if wav.shape[1] < n_fft:
        wav = F.pad(wav, (0, n_fft - wav.shape[1]))
    mel_basis = get_mel_filters(sr, n_fft, n_mels, fmin, fmax).to(wav.device)
    hann_win = get_hann_window(win_size).to(wav.device)
    spec = torch.stft(wav.squeeze(0), n_fft, hop_length=hop_size, win_length=win_size, window=hann_win, return_complex=True)
    spec = torch.abs(spec)
    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = mel.transpose(0, 1).unsqueeze(0)
    return mel


def extract_embeddings_for_split(DATA_DIR, EMBEDDINGS_DIR, split_name, qwen3_encoder, device, max_utterances_per_speaker=50):
    split_path = os.path.join(DATA_DIR, split_name)
    output_file = os.path.join(EMBEDDINGS_DIR, f"{split_name}_embeddings.safetensors")
    meta_file = os.path.join(EMBEDDINGS_DIR, f"{split_name}_meta.json")
    if os.path.exists(output_file):
        print(f"{split_name} already extracted")
        return output_file, meta_file
    all_mels, all_embeddings, meta = [], [], []
    speaker_dirs = sorted(os.listdir(split_path))
    print(f"Processing {len(speaker_dirs)} speakers...")
    for si, speaker_id in enumerate(speaker_dirs):
        speaker_path = os.path.join(split_path, speaker_id)
        if not os.path.isdir(speaker_path): continue
        utt_count = 0
        for book_id in sorted(os.listdir(speaker_path)):
            book_path = os.path.join(speaker_path, book_id)
            if not os.path.isdir(book_path): continue
            for fname in sorted(os.listdir(book_path)):
                if not fname.endswith('.flac') or utt_count >= max_utterances_per_speaker: continue
                try:
                    mel = extract_mel(os.path.join(book_path, fname))
                    with torch.no_grad():
                        mel_dev = mel.to(device).to(torch.bfloat16 if device.type=='cuda' else torch.float32)
                        gt_emb = qwen3_encoder(mel_dev).float().cpu()
                    all_mels.append(mel.squeeze(0).cpu())
                    all_embeddings.append(gt_emb.squeeze(0))
                    meta.append({'speaker':speaker_id,'file':fname,'mel_len':mel.shape[1]})
                    utt_count += 1
                except Exception as e: print(f'  Skip {fname}: {e}')
        if (si+1)%20==0: print(f'  {si+1}/{len(speaker_dirs)} done')
    save_file({'embeddings':torch.stack(all_embeddings)}, output_file)
    mel_dir = os.path.join(EMBEDDINGS_DIR, f'{split_name}_mels')
    os.makedirs(mel_dir, exist_ok=True)
    for i, mel in enumerate(all_mels): torch.save(mel, os.path.join(mel_dir, f'{i:06d}.pt'))
    with open(meta_file,'w') as f: json.dump(meta, f)
    print(f'Saved {len(all_embeddings)} embeddings')
    return output_file, meta_file

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, embeddings_file, mel_dir, max_mel_len=400):
        self.embeddings = load_file(embeddings_file)['embeddings']
        self.mel_dir = mel_dir
        self.max_mel_len = max_mel_len
        self.n_samples = self.embeddings.shape[0]
        print(f'Loaded {self.n_samples} samples')
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        mel = torch.load(os.path.join(self.mel_dir, f'{idx:06d}.pt'), weights_only=True)
        T = mel.shape[0]
        if T > self.max_mel_len:
            start = torch.randint(0, T - self.max_mel_len, (1,)).item()
            mel = mel[start:start+self.max_mel_len]
        elif T < self.max_mel_len:
            mel = F.pad(mel, (0, 0, 0, self.max_mel_len - T))
        return mel, self.embeddings[idx]
    

def get_train_test_loaders():
    DATA_DIR = "./data/LibriSpeech"
    EMBEDDINGS_DIR = "./data/embeddings"
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("Loading Qwen3-TTS model...")
    qwen3_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0" if device.type == "cuda" else "cpu", dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    qwen3_encoder = qwen3_model.model.speaker_encoder.eval()
    print(f"Qwen3 speaker encoder loaded on {device}")

    # Generate/Load the embedding split files
    train_emb_file, train_meta_file = extract_embeddings_for_split(DATA_DIR, EMBEDDINGS_DIR, "train-clean-100", qwen3_encoder, device)
    dev_emb_file, dev_meta_file = extract_embeddings_for_split(DATA_DIR, EMBEDDINGS_DIR, "dev-clean", qwen3_encoder, device)

    BATCH_SIZE = 32
    MAX_MEL_LEN = 400
    train_dataset = SpeakerEmbeddingDataset(train_emb_file, os.path.join(EMBEDDINGS_DIR, 'train-clean-100_mels'), MAX_MEL_LEN)
    dev_dataset = SpeakerEmbeddingDataset(dev_emb_file, os.path.join(EMBEDDINGS_DIR, 'dev-clean_mels'), MAX_MEL_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f'Train: {len(train_dataset)} samples, {len(train_loader)} batches')
    print(f'Test:  {len(dev_dataset)} samples, {len(test_loader)} batches')

    return train_loader, test_loader