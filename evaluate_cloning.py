import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from jiwer import wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import (
    BasicSpeakerEncoder,
    LightweightECAPATDNN,
    TDNNSpeakerEncoder,
    ConvEncoder,
)
import numpy as np
import os
import random
import re
import string
import time
from torchview import draw_graph
from torchinfo import summary


def fit_audio_length(audio_array, target_num_samples):
    if len(audio_array) > target_num_samples:
        return audio_array[:target_num_samples]
    if len(audio_array) < target_num_samples:
        pad_width = target_num_samples - len(audio_array)
        return np.pad(audio_array, (0, pad_width))
    return audio_array


def calculate_wer(audio_array, target_text, processor, asr_model, sr=24000, device="cuda:0", max_words=None):
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
    
    if max_words is not None:
        transcription_words = transcription.split()
        transcription = " ".join(transcription_words[:max_words])

    # Clean punctuation for a fairer WER calculation
    clean_target = re.sub(f"[{re.escape(string.punctuation)}]", "", target_text.lower())
    clean_transcription = re.sub(f"[{re.escape(string.punctuation)}]", "", transcription.lower())

    # Calculate Word Error Rate (lower is better)
    error_rate = wer(clean_target, clean_transcription)
    return error_rate, transcription

def evaluate_cloning(custom_speaker_encoder=False, model=None, model_name="BasicSpeakerEncoder", model_path="checkpoints/BasicSpeakerEncoder_trial_3/best.pt", number_of_speakers=5):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
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

        # Create a diagram of the custom speaker encoder architecture for the report
        os.makedirs(f"./data/custom_model_diagram/{model_name}", exist_ok=True)
        dummy_input = torch.randn((1, 400, 128), dtype=tts.model.dtype, device=tts.device)
        # Update the depth based on how complex the model is
        model_graph = draw_graph(custom_encoder, input_data=dummy_input, depth=1)
        model_graph.visual_graph.render(format='pdf', filename=f"./data/custom_model_diagram/{model_name}/{model_name}_architecture", cleanup=True)
        summary(custom_encoder, input_data=dummy_input)

    # Save base model diagram for reference
    os.makedirs(f"./data/custom_model_diagram/Qwen3_Original_SpeakerEncoder", exist_ok=True)
    dummy_input = torch.randn((1, 400, 128), dtype=tts.model.dtype, device=tts.device)
    base_graph = draw_graph(tts.model.speaker_encoder, input_data=dummy_input, depth=1)
    base_graph.visual_graph.render(format='pdf', filename=f"./data/custom_model_diagram/Qwen3_Original_SpeakerEncoder/Qwen3_Original_SpeakerEncoder_architecture", cleanup=True)
    summary(tts.model.speaker_encoder, input_data=dummy_input)
    
    
    # 3. Evaluate across a subset of test-clean
    data_dir = "./data/LibriSpeech/test-clean"
    speakers = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    
    total_wer = 0.0
    total_sim = 0.0
    eval_count = 0
    
    elapsed_time = 0
    start_time = time.time()
    while eval_count < number_of_speakers:
        speaker = random.choice(speakers)
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

            if len(ref_utt["text"].split()) < 10 or len(target_utt["text"].split()) < 10:
                continue  # Skip very short utterances
            
            target_text = target_utt["text"]
            target_audio_path = target_utt["audio"]
            ref_audio = ref_utt["audio"]
            ref_text = ref_utt["text"]
            
            print(f"\n--- Evaluating Speaker {speaker} ---")
            print(f"Ref Audio  : {ref_audio}")
            print(f"Target Text: {target_text}")
            
            tgt_wav, orig_sr = torchaudio.load(target_audio_path)
            if orig_sr != 24000:
                tgt_wav = torchaudio.functional.resample(tgt_wav, orig_sr, 24000)
            tgt_audio_array = tgt_wav.mean(dim=0).numpy()

            # 4. Generate Audio
            print("Generating Cloned Audio...")
            wavs, sr = tts.generate_voice_clone(
                text=target_text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text
            )
            gen_audio = wavs[0]

            target_num_samples = int(round(len(tgt_audio_array) * sr / 24000))
            gen_audio = fit_audio_length(gen_audio, target_num_samples)

            # Save generated audio for inspection
            os.makedirs(f"./data/generated_samples_custom_model_name_{model_name}", exist_ok=True)
            sf.write(f"./data/generated_samples_custom_model_name_{model_name}/{speaker}_{os.path.basename(target_audio_path)}", gen_audio, sr)
            print(f"✅ Success! Audio saved as generated_samples_custom_model_name_{model_name}/{speaker}_{os.path.basename(target_audio_path)}")

            # 5. Evaluate Speaker Cosine Similarity
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
            wer_score, transcript = calculate_wer(
                gen_audio,
                target_text,
                processor,
                asr_model,
                sr=sr,
                device=device,
                max_words=len(target_text.split()),
            )
            
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
    # # Baseline with Qwen3's original speaker encoder
    # print("Evaluating baseline with Qwen3's original speaker encoder...")
    # evaluate_cloning(custom_speaker_encoder=False, model=None, model_name="Qwen3_Original_SpeakerEncoder", model_path=None, number_of_speakers=1)

    # # Basic Speaker Encoder
    # model = BasicSpeakerEncoder()
    # print("Evaluating Basic Speaker Encoder...")
    # evaluate_cloning(custom_speaker_encoder=True, model=model, model_name="BasicSpeakerEncoder", model_path="final_weights/BasicSpeakerEncoder/best.pt", number_of_speakers=1)

    # # Song's ConvEncoder
    # model = ConvEncoder()
    # print("Evaluating Conv Encoder...")
    # evaluate_cloning(custom_speaker_encoder=True, model=model, model_name="ConvEncoder", model_path="final_weights/ConvEncoder/best.pt", number_of_speakers=1)

    # # Rohan's verified ECAPA-TDNN encoder
    # model = TDNNSpeakerEncoder()
    # print("Evaluating TDNN Speaker Encoder...")
    # evaluate_cloning(custom_speaker_encoder=True, model=model, model_name="TDNNSpeakerEncoder", model_path="final_weights/TDNNSpeakerEncoder/best.pt", number_of_speakers=1)

    # Aditya's lightweight ECAPA-TDNN
    model = LightweightECAPATDNN(enc_dim=2048)
    print("Evaluating Lightweight ECAPA-TDNN...")
    evaluate_cloning(
        custom_speaker_encoder=True,
        model=model,
        model_name="LightweightECAPA_TDNNSpeakerEncoder",
        model_path="final_weights/LightweightECAPA_TDNNSpeakerEncoder/best.pt",
        number_of_speakers=100,
    )
