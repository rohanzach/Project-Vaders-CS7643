import torch
import time
import json
import torch.nn.functional as F
import os
from extract_embedding import get_train_test_loaders
from qwen_tts.core.models import BasicSpeakerEncoder

def distillation_loss(pred_emb, gt_emb, alpha=0.7):
    cosine_loss = 1.0 - F.cosine_similarity(pred_emb, gt_emb, dim=-1).mean()
    mse_loss = F.mse_loss(pred_emb, gt_emb)
    return alpha * cosine_loss + (1-alpha) * mse_loss, cosine_loss.item(), mse_loss.item()

def training_loop(model, train_loader, test_loader, device, hyperparams):
    os.makedirs("checkpoints", exist_ok=True)
    CHECKPOINT_DIR = "checkpoints"

    NUM_EPOCHS = hyperparams.get('num_epochs', 50)
    SAVE_EVERY = hyperparams.get('save_every', 10)
    LR = hyperparams.get('lr', 1e-3)
    WEIGHT_DECAY = hyperparams.get('weight_decay', 1e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_test_cosine = 0.0
    history = {'train_loss':[],'train_cosine':[],'test_loss':[],'test_cosine':[]}
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        epoch_loss, epoch_cos, n_batches = 0, 0, 0
        t0 = time.time()
        for mel, gt_emb in train_loader:
            mel, gt_emb = mel.to(device), gt_emb.to(device)
            pred_emb = model(mel)
            loss, cos_l, mse_l = distillation_loss(pred_emb, gt_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_cos += (1.0 - cos_l)
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / n_batches
        train_cosine = epoch_cos / n_batches
        model.eval()
        dev_loss, dev_cos, dev_n = 0, 0, 0
        with torch.no_grad():
            for mel, gt_emb in test_loader:
                mel, gt_emb = mel.to(device), gt_emb.to(device)
                pred_emb = model(mel)
                loss, cos_l, _ = distillation_loss(pred_emb, gt_emb)
                dev_loss += loss.item()
                dev_cos += (1.0 - cos_l)
                dev_n += 1
        dev_loss /= max(dev_n,1)
        dev_cosine = dev_cos / max(dev_n,1)
        dt = time.time() - t0
        history['train_loss'].append(train_loss)
        history['train_cosine'].append(train_cosine)
        history['test_loss'].append(dev_loss)
        history['test_cosine'].append(dev_cosine)
        marker = ''
        if dev_cosine > best_test_cosine:
            best_test_cosine = dev_cosine
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best.pt'))
            marker = ' *best*'
        print(f'Epoch {epoch:3d}/{NUM_EPOCHS} | Train loss={train_loss:.4f} cos={train_cosine:.4f} | Test loss={dev_loss:.4f} cos={dev_cosine:.4f}{marker} | {dt:.1f}s')
        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'epoch_{epoch}.pt'))
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final.pt'))
    with open(os.path.join(CHECKPOINT_DIR, 'history.json'), 'w') as f: json.dump(history, f)
    print(f'\nDone! Best test cosine similarity: {best_test_cosine:.4f}')
