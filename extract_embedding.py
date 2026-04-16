import torch
import librosa
from qwen_tts import Qwen3TTSModel
import os
from safetensors.torch import save_file


@torch.inference_mode()
def create_speaker_embeddings(audio_paths="./audio_data/LibriSpeech/dev-clean/"):
    
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16
    )

    # Loop through each folder in the directory (Each folder is a person)
    for person_folder in os.listdir(audio_paths):
        person_path = os.path.join(audio_paths, person_folder)
        
        if os.path.isdir(person_path):

            speaker_embeddings = []
            print(f"Processing speaker: {person_folder}...")
            
            for root, _, files in os.walk(person_path):
                for file_name in files:
                    if file_name.endswith(('.flac')):
                        file_path = os.path.join(root, file_name)
                        audio, sr = librosa.load(file_path, sr=24000)
                        
                        # Access the method from the underlying inner model
                        speaker_embedding = model.model.extract_speaker_embedding(audio, sr)
                        speaker_embeddings.append(speaker_embedding)

            # Write the speaker embeddings to a file
            cpu_embeddings = [emb.cpu() for emb in speaker_embeddings]
            embeddings_tensor = torch.stack(cpu_embeddings) if cpu_embeddings[0].dim() == 1 else torch.cat(cpu_embeddings, dim=0)
            
            output_file = os.path.join(person_path, f"{person_folder}_embeddings.safetensors")
            save_file({"embeddings": embeddings_tensor}, output_file)
            print(f"Saved {len(speaker_embeddings)} embeddings to {output_file}")

if __name__ == "__main__":
    create_speaker_embeddings()