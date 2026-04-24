import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from jiwer import wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import BasicSpeakerEncoder
import numpy as np
import os
import random
import re
import string
import time


def calculate_wer(audio_array, target_text, processor, asr_model, sr=24000, device="cuda:0"):
    # Whisper expects 16kHz audio
    if sr != 16000:
        audio_tensor = torch.tensor(audio_array).unsqueeze(0).float()
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sr, new_freq=16000)
        audio_array = audio_tensor.squeeze(0).numpy()
        
    inputs = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt", 
        return_attention_mask=True
    ).to(device)
    
    with torch.no_grad():
        predicted_ids = asr_model.generate(
            inputs.input_features,
            attention_mask=inputs.get("attention_mask"),
            pad_token_id=asr_model.config.eos_token_id
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Clean punctuation for a fairer WER calculation
    clean_target = re.sub(f"[{re.escape(string.punctuation)}]", "", target_text.lower())
    clean_transcription = re.sub(f"[{re.escape(string.punctuation)}]", "", transcription.lower())

    # Calculate Word Error Rate (lower is better)
    error_rate = wer(clean_target, clean_transcription)
    return error_rate, transcription

def evaluate_cloning(custom_speaker_encoder=False, model=None, model_path="checkpoints/BasicSpeakerEncoder_trial_3/best.pt", number_of_speakers=5):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Evaluation Models
    print("Loading Whisper for WER evaluation...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(device)
    
    # 2. Load TTS Model and Custom Encoder
    print("Loading TTS Pipeline...")
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=device, dtype=torch.bfloat16)
    
    if custom_speaker_encoder:
        # Save the original base speaker encoder to use for evaluation
        base_speaker_encoder = tts.model.speaker_encoder
    
        print("Use custom speaker encoder for evaluation...")
        custom_encoder = model
        if custom_encoder is None:
            custom_encoder = BasicSpeakerEncoder()
        custom_encoder.load_state_dict(torch.load(model_path, map_location=device))
        custom_encoder.to(tts.device).to(tts.model.dtype).eval()
        tts.model.speaker_encoder = custom_encoder
    
    # 3. Evaluate across a subset of test-clean
    data_dir = "./data/LibriSpeech/test-clean"
    speakers = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Limit to a few random speakers to save time
    num_speakers = min(number_of_speakers, len(speakers))
    speakers = random.sample(speakers, num_speakers)
    
    total_wer = 0.0
    total_sim = 0.0
    eval_count = 0
    
    elapsed_time = 0
    start_time = time.time()
    for speaker in speakers:
        speaker_dir = os.path.join(data_dir, speaker)
        utterances = []
        
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if not os.path.isdir(chapter_dir): continue
            
            trans_file = os.path.join(chapter_dir, f"{speaker}-{chapter}.trans.txt")
            if not os.path.exists(trans_file): continue
                
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        audio_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                        if os.path.exists(audio_path):
                            utterances.append({"audio": audio_path, "text": text})
                            
        if len(utterances) >= 2:
            # Randomly select a reference utterance and a target utterance
            ref_utt, target_utt = random.sample(utterances, 2)
            
            target_text = target_utt["text"]
            target_audio_path = target_utt["audio"]
            ref_audio = ref_utt["audio"]
            ref_text = ref_utt["text"]
            
            print(f"\n--- Evaluating Speaker {speaker} ---")
            print(f"Ref Audio  : {ref_audio}")
            print(f"Target Text: {target_text}")
            
            # 4. Generate Audio
            print("Generating Cloned Audio...")
            wavs, sr = tts.generate_voice_clone(
                text=target_text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text
            )
            gen_audio = wavs[0]
            
            # 5. Evaluate Speaker Cosine Similarity
            tgt_wav, orig_sr = torchaudio.load(target_audio_path)
            if orig_sr != 24000:
                tgt_wav = torchaudio.functional.resample(tgt_wav, orig_sr, 24000)
            tgt_audio_array = tgt_wav.mean(dim=0).numpy()
            
            # If we are using a custom speaker encoder, we should evaluate using the Qwen3 speaker encoder
            if custom_speaker_encoder:
                # Temporarily swap back to the base encoder to measure cosine similarity
                tts.model.speaker_encoder = base_speaker_encoder
            
            emb_gen = tts.model.extract_speaker_embedding(gen_audio, sr)
            emb_tgt = tts.model.extract_speaker_embedding(tgt_audio_array, 24000)
            
            if custom_speaker_encoder:
                # Restore your custom encoder for the next generation
                tts.model.speaker_encoder = custom_encoder
            
            cos_sim = F.cosine_similarity(emb_gen.unsqueeze(0), emb_tgt.unsqueeze(0)).item()
            
            # 6. Evaluate WER
            wer_score, transcript = calculate_wer(gen_audio, target_text, processor, asr_model, sr=sr, device=device)
            
            print(f"Transcription: {transcript.strip()}")
            print(f"WER (Words)  : {wer_score * 100:.2f}%")
            print(f"Cosine Sim   : {cos_sim:.4f}")
            
            total_wer += wer_score
            total_sim += cos_sim
            eval_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    if eval_count > 0:
        print(f"\n=== Final Results ===")
        print(f"Total Evaluation Time: {elapsed_time:.2f} seconds")
        print(f"Average WER across {eval_count} speakers: {(total_wer / eval_count) * 100:.2f}%")
        print(f"Average Cosine Sim across {eval_count} speakers: {total_sim / eval_count:.4f}")

    return total_wer, total_sim, eval_count

if __name__ == "__main__":
    evaluate_cloning(custom_speaker_encoder=False, model_path="checkpoints/BasicSpeakerEncoder_trial_3/best.pt", number_of_speakers=20)