import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from vq_vae import EEG_FSQ_VAE
import os
import time

# --- НАСТРОЙКИ ---
DATA_FILE = "eeg_dataset_4ch_shared.pt"
WEIGHTS_FILE = "fsq_vae_4ch_best.weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры модели (должны совпадать с train_4ch_fsq.py)
INPUT_DIM = 128
HIDDEN_DIM = 256
FSQ_LEVELS = [2] * 16

def run_validation():
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Файл весов {WEIGHTS_FILE} пока не создан. Ждем...")
        return

    print(f"--- Валидация весов: {WEIGHTS_FILE} ---")
    
    # Загружаем данные (берем только начало для скорости)
    try:
        data_tensor = torch.load(DATA_FILE)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return

    # Берем случайный батч
    test_loader = DataLoader(TensorDataset(data_tensor), batch_size=100, shuffle=True)
    x_test_batch = next(iter(test_loader))[0].to(DEVICE)

    # Инициализируем модель
    model = EEG_FSQ_VAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        levels=FSQ_LEVELS
    ).to(DEVICE)

    # Загружаем веса (сразу пробуем загрузить, если файл в процессе записи - может быть ошибка, обработаем её)
    try:
        state_dict = torch.load(WEIGHTS_FILE, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Не удалось прочитать веса (возможно, они записываются): {e}")
        return
        
    model.eval()

    with torch.no_grad():
        _, x_recon_batch, indices = model(x_test_batch)
        mse = F.mse_loss(x_recon_batch, x_test_batch).item()
        used_tokens = len(torch.unique(indices))

    print(f"MSE: {mse:.6f} | Used Tokens in batch: {used_tokens}")

    # Рисуем
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()
    
    for i in range(30):
        orig = x_test_batch[i].cpu().numpy()
        recon = x_recon_batch[i].cpu().numpy()
        
        axes[i].plot(orig, label="Orig", color='blue', alpha=0.5)
        axes[i].plot(recon, label="Recon", color='red', linestyle='--')
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(True, alpha=0.2)
        axes[i].set_title(f"Sample {i}")

    plt.suptitle(f"FSQ-VAE Validation\nMSE: {mse:.6f} | Used: {used_tokens}/4096\nFile: {WEIGHTS_FILE}", fontsize=16)
    plt.tight_layout()
    
    output_png = "live_fsq_val.png"
    plt.savefig(output_png)
    plt.close()
    print(f"График обновлен: {output_png}")

if __name__ == "__main__":
    run_validation()
