import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from vq_vae import EEG_FSQ_VAE 
import sys
import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- НАСТРОЙКИ ---
DATA_FILE = "eeg_dataset_4ch_shared.pt"
INPUT_DIM = 128         
BATCH_SIZE = 4096       
MAX_EPOCHS = 200        
LEARNING_RATE = 5e-4    
WEIGHT_DECAY = 1e-5     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Уровни для получения 65536 токенов (2^16)
FSQ_LEVELS = [2] * 16

def main():
    log_filename = f"train_4ch_fsq_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout

    print(f"Running FSQ-VAE on: {DEVICE}")
    print(f"Levels: {FSQ_LEVELS} (Total tokens: {np.prod(FSQ_LEVELS)})")
    
    print(f"Loading {DATA_FILE}...")
    try:
        data_tensor = torch.load(DATA_FILE)
    except FileNotFoundError:
        print("Файл не найден!")
        return

    dataset_size = len(data_tensor)
    val_size = min(50000, int(dataset_size * 0.05)) 
    train_size = dataset_size - val_size
    train_ds, val_ds = random_split(TensorDataset(data_tensor), [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EEG_FSQ_VAE(
        input_dim=INPUT_DIM, 
        hidden_dim=256,
        levels=FSQ_LEVELS
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_recon = float('inf')
    
    print(f"Начинаем обучение FSQ-VAE...")
    
    try:
        for epoch in range(MAX_EPOCHS):
            model.train()
            train_recon_acc = 0
            
            for batch in train_loader:
                x = batch[0].to(DEVICE)
                
                _, x_recon, _ = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                loss = recon_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_recon_acc += recon_loss.item()
                
            # Валидация
            model.eval()
            val_recon_acc = 0
            all_indices = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(DEVICE)
                    _, x_recon, indices = model(x)
                    val_recon_acc += F.mse_loss(x_recon, x).item()
                    all_indices.extend(indices.cpu().numpy())
            
            avg_val_recon = val_recon_acc / len(val_loader)
            n_tr = len(train_loader)
            
            # Считаем использование словаря (просто для интереса, в FSQ он не дохнет)
            unique_tokens = len(np.unique(all_indices))
            
            print(f"Epoch {epoch+1:2d} | Recon: {train_recon_acc/n_tr:.6f} | Val Recon: {avg_val_recon:.6f} | Used: {unique_tokens}")

            # Сохранение чекпоинта
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), "fsq_vae_4ch_checkpoint.weights")

            # Сохранение лучшей модели
            if avg_val_recon < best_val_recon:
                best_val_recon = avg_val_recon
                torch.save(model.state_dict(), "fsq_vae_4ch_best.weights")
                print(f"--- Новое лучшее состояние: {best_val_recon:.6f} ---")

    except KeyboardInterrupt:
        print("\n[ОСТАНОВКА] Прервано пользователем.")
    
    torch.save(model.state_dict(), "fsq_vae_4ch_final.weights")
    print("\nОбучение завершено.")
    
    # Визуализация
    model.load_state_dict(torch.load("fsq_vae_4ch_best.weights"))
    model.eval()
    x_val_batch = next(iter(val_loader))[0].to(DEVICE)
    with torch.no_grad():
        _, x_recon_batch, _ = model(x_val_batch)
    
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()
    indices = np.random.choice(len(x_val_batch), 30, replace=False)
    for i, idx in enumerate(indices):
        orig, recon = x_val_batch[idx].cpu().numpy(), x_recon_batch[idx].cpu().numpy()
        axes[i].plot(orig, color='blue', alpha=0.5)
        axes[i].plot(recon, color='red', linestyle='--')
        axes[i].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("fsq_results_4ch.png")
    print("Результаты сохранены в fsq_results_4ch.png")

if __name__ == "__main__":
    main()
