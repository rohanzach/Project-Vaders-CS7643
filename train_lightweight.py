import os
import shutil
import torch

from training_loop import training_loop
from extract_embedding import get_train_test_loaders
from qwen_tts.core.models import LightweightECAPATDNN


MODEL_NAME = "LightweightECAPA_TDNNSpeakerEncoder"


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training device: {device}")

    train_loader, test_loader = get_train_test_loaders()

    model = LightweightECAPATDNN(enc_dim=2048).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{MODEL_NAME}: {n_params:,} params")

    hyperparams = {
        "num_epochs": 20,
        "save_every": 10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "loss_alpha": 0.7,
        "model_name": MODEL_NAME,
    }

    best_cos = training_loop(model, train_loader, test_loader, device, hyperparams)
    print(f"Best test cosine: {best_cos:.4f}")

    src = os.path.join("checkpoints", MODEL_NAME, "best.pt")
    dst_dir = os.path.join("final_weights", MODEL_NAME)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, "best.pt"))
    print(f"Copied {src} -> {dst_dir}/best.pt")


if __name__ == "__main__":
    main()
