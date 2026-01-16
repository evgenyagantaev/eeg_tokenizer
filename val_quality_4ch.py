import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from vq_vae import EEG_VQ_VAE

# --- НАСТРОЙКИ ---
DATA_FILE = "eeg_dataset_4ch_shared.pt"
WEIGHTS_FILE = "vq_vae_4ch.weights"  # Поменяйте на vq_vae_4ch_final.weights если нужно
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры модели (должны совпадать с train_4ch.py)
INPUT_DIM = 128
HIDDEN_DIM = 256
EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 2048

def main():
    print(f"Загрузка данных из {DATA_FILE}...")
    try:
        data_tensor = torch.load(DATA_FILE)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return

    print(f"Размер датасета: {data_tensor.shape}")
    
    # Берем случайную выборку для проверки
    test_loader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=True)
    x_test_batch = next(iter(test_loader))[0].to(DEVICE)

    print(f"Загрузка модели и весов: {WEIGHTS_FILE}...")
    model = EEG_VQ_VAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_embeddings=NUM_EMBEDDINGS
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")
        return
        
    model.eval()

    print("Инференс (восстановление спектров)...")
    with torch.no_grad():
        vq_loss, x_recon_batch, _ = model(x_test_batch)
        recon_mse = F.mse_loss(x_recon_batch, x_test_batch).item()

    print(f"MSE на этой выборке: {recon_mse:.6f}")

    # Визуализация
    print("Генерация графиков...")
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()
    
    # Выбираем 30 случайных примеров из батча
    indices = np.random.choice(len(x_test_batch), 30, replace=False)
    
    for i, idx in enumerate(indices):
        orig = x_test_batch[idx].cpu().numpy()
        recon = x_recon_batch[idx].cpu().numpy()
        
        axes[i].plot(orig, label="Orig", color='blue', alpha=0.6)
        axes[i].plot(recon, label="Recon", color='red', linestyle='--', alpha=0.8)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_title(f"Sample {idx}")
        axes[i].grid(True, alpha=0.2)
        if i == 0:
            axes[i].legend()

    plt.suptitle(f"Реконструкция 4-канальных токенов (MSE: {recon_mse:.6f})\nWeights: {WEIGHTS_FILE}", fontsize=16)
    plt.tight_layout()
    
    output_png = "validation_results_4ch.png"
    plt.savefig(output_png)
    print(f"Готово! График сохранен в {output_png}")

if __name__ == "__main__":
    main()
