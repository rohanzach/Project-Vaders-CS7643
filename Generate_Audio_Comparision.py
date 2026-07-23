import argparse
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple

import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import (
	BasicSpeakerEncoder,
	ConvEncoder,
	LightweightECAPATDNN,
	TDNNSpeakerEncoder,
)


def resolve_checkpoint_path(path_value: str) -> str:
	"""Resolve checkpoint path from either a .pt file path or a directory.

	Directory resolution order: best.pt -> final.pt -> single *.pt file.
	"""
	if not path_value:
		raise ValueError("Checkpoint path is empty.")

	if os.path.isfile(path_value):
		return path_value

	if not os.path.isdir(path_value):
		raise FileNotFoundError(f"Checkpoint path does not exist: {path_value}")

	best_path = os.path.join(path_value, "best.pt")
	if os.path.isfile(best_path):
		return best_path

	final_path = os.path.join(path_value, "final.pt")
	if os.path.isfile(final_path):
		return final_path

	pt_files = [f for f in os.listdir(path_value) if f.endswith(".pt")]
	if len(pt_files) == 1:
		return os.path.join(path_value, pt_files[0])

	if len(pt_files) > 1:
		raise FileNotFoundError(
			f"Multiple .pt files found in {path_value}. Expected best.pt/final.pt or exactly one .pt file."
		)

	raise FileNotFoundError(
		f"No checkpoint file found in {path_value}. Expected best.pt/final.pt or a .pt file."
	)


def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def load_state_dict_safely(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
	ckpt = torch.load(checkpoint_path, map_location=device)
	if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
		state_dict = ckpt["model_state_dict"]
	else:
		state_dict = ckpt
	model.load_state_dict(state_dict, strict=True)


def collect_speaker_utterances(speaker_dir: str, speaker_id: str) -> List[Dict[str, str]]:
	utterances: List[Dict[str, str]] = []
	for chapter in sorted(os.listdir(speaker_dir)):
		chapter_dir = os.path.join(speaker_dir, chapter)
		if not os.path.isdir(chapter_dir):
			continue

		trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter}.trans.txt")
		if not os.path.exists(trans_file):
			continue

		with open(trans_file, "r", encoding="utf-8") as f:
			for line in f:
				parts = line.strip().split(" ", 1)
				if len(parts) != 2:
					continue
				utt_id, text = parts
				audio_path = os.path.join(chapter_dir, f"{utt_id}.flac")
				if os.path.exists(audio_path) and len(text.split()) >= 10:
					utterances.append({"utt_id": utt_id, "audio": audio_path, "text": text})
	return utterances


def choose_eval_pairs(data_dir: str, num_speakers: int, seed: int) -> List[Dict[str, str]]:
	random.seed(seed)
	all_speakers = [
		s for s in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, s))
	]

	random.shuffle(all_speakers)
	selected: List[Dict[str, str]] = []

	for speaker in all_speakers:
		speaker_dir = os.path.join(data_dir, speaker)
		utterances = collect_speaker_utterances(speaker_dir, speaker)
		if len(utterances) < 2:
			continue

		ref_utt, target_utt = random.sample(utterances, 2)
		selected.append(
			{
				"speaker": speaker,
				"ref_audio": ref_utt["audio"],
				"ref_text": ref_utt["text"],
				"target_audio": target_utt["audio"],
				"target_text": target_utt["text"],
				"target_utt": target_utt["utt_id"],
			}
		)
		if len(selected) == num_speakers:
			break

	if len(selected) < num_speakers:
		raise RuntimeError(
			f"Only found {len(selected)} speakers with valid utterance pairs, requested {num_speakers}."
		)

	return selected


def save_original_audios(eval_pairs: List[Dict[str, str]], output_dir: str) -> None:
	original_dir = os.path.join(output_dir, "original")
	os.makedirs(original_dir, exist_ok=True)

	for item in eval_pairs:
		speaker = item["speaker"]
		target_utt = item["target_utt"]
		target_audio = item["target_audio"]
		destination = os.path.join(original_dir, f"{speaker}_{target_utt}.flac")
		shutil.copy2(target_audio, destination)
		print(f"Saved original: {destination}")


def instantiate_custom_encoder(
	name: str,
	checkpoint_path: Optional[str],
	model_dtype: torch.dtype,
	model_device: torch.device,
) -> torch.nn.Module:
	if name == "basic":
		encoder = BasicSpeakerEncoder()
	elif name == "custom_ecapa_tdnn":
		encoder = TDNNSpeakerEncoder()
	elif name == "custom_conv_glub":
		encoder = ConvEncoder()
	elif name == "qwen3_base":
		# This checkpoint uses a 2048-d output projection head.
		encoder = LightweightECAPATDNN(enc_dim=2048)
	else:
		raise ValueError(f"Unknown encoder name: {name}")

	if checkpoint_path:
		resolved_path = resolve_checkpoint_path(checkpoint_path)
		load_state_dict_safely(encoder, resolved_path, model_device)

	encoder.to(model_device).to(model_dtype).eval()
	return encoder


def generate_for_encoder(
	tts: Qwen3TTSModel,
	encoder_name: str,
	eval_pairs: List[Dict[str, str]],
	output_dir: str,
	checkpoint_path: Optional[str] = None,
) -> None:
	print(f"\n=== Generating with encoder: {encoder_name} ===")

	original_encoder = tts.model.speaker_encoder
	custom_encoder = instantiate_custom_encoder(
		name=encoder_name,
		checkpoint_path=checkpoint_path,
		model_dtype=tts.model.dtype,
		model_device=tts.device,
	)
	tts.model.speaker_encoder = custom_encoder

	encoder_output_dir = os.path.join(output_dir, encoder_name)
	os.makedirs(encoder_output_dir, exist_ok=True)

	try:
		for item in eval_pairs:
			speaker = item["speaker"]
			target_text = item["target_text"]
			ref_audio = item["ref_audio"]
			ref_text = item["ref_text"]
			target_utt = item["target_utt"]

			print(f"[{encoder_name}] Speaker {speaker} | target_utt={target_utt}")

			wavs, sr = tts.generate_voice_clone(
				text=target_text,
				language="English",
				ref_audio=ref_audio,
				ref_text=ref_text,
			)

			out_path = os.path.join(encoder_output_dir, f"{speaker}_{target_utt}.flac")
			sf.write(out_path, wavs[0], sr)
			print(f"Saved: {out_path}")
	finally:
		tts.model.speaker_encoder = original_encoder


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate cloned audio for 5 test-clean speakers using 4 encoder models."
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default="./data/LibriSpeech/test-clean",
		help="Root directory containing LibriSpeech speaker folders.",
	)
	parser.add_argument(
		"--num-speakers",
		type=int,
		default=5,
		help="How many speakers to sample from test-clean.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducible speaker sampling.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="./data/generated_audio_comparison",
		help="Output directory for generated audio.",
	)
	parser.add_argument(
		"--basic-ckpt",
		type=str,
		default="final_weights/basicspeakerencoder",
		help="Checkpoint for BasicSpeakerEncoder.",
	)
	parser.add_argument(
		"--custom-ecapa-ckpt",
		type=str,
		default="final_weights/TDNNSpeakerEncoder",
		help="Checkpoint for Custom_ECAPA_TDNN encoder.",
	)
	parser.add_argument(
		"--custom-conv-ckpt",
		type=str,
		default="final_weights/ConvEncoder",
		help="Checkpoint for Custom_Conv_GLUB encoder.",
	)
	parser.add_argument(
		"--qwen3-base-ckpt",
		type=str,
		default="final_weights/LightWeightECAPA_TDNNSpeakerEncoder",
		help="Checkpoint for Qwen3_base (LightweightECAPATDNN) encoder.",
	)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	device = get_device()
	print(f"Using device: {device}")

	eval_pairs = choose_eval_pairs(args.data_dir, args.num_speakers, args.seed)
	print("Selected speakers:", [p["speaker"] for p in eval_pairs])
	save_original_audios(eval_pairs, args.output_dir)

	tts = Qwen3TTSModel.from_pretrained(
		"Qwen/Qwen3-TTS-12Hz-1.7B-Base",
		device_map=device,
		dtype=torch.bfloat16,
	)

	checkpoint_map = {
		"basic": args.basic_ckpt or None,
		"custom_ecapa_tdnn": args.custom_ecapa_ckpt or None,
		"custom_conv_glub": args.custom_conv_ckpt or None,
		"qwen3_base": args.qwen3_base_ckpt or None,
	}

	for encoder_name in ["qwen3_base", "custom_ecapa_tdnn", "custom_conv_glub", "basic"]:
		try:
			generate_for_encoder(
				tts=tts,
				encoder_name=encoder_name,
				eval_pairs=eval_pairs,
				output_dir=args.output_dir,
				checkpoint_path=checkpoint_map[encoder_name],
			)
		except Exception as exc:
			print(f"Skipping {encoder_name} due to error: {exc}")

	print("\nDone.")
	print(f"Generated files are in: {args.output_dir}")


if __name__ == "__main__":
	main()
