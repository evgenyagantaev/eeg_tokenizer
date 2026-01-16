import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from vq_vae import EEG_VQ_VAE 
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
MAX_EPOCHS = 500        
LEARNING_RATE = 1e-4    
WEIGHT_DECAY = 1e-5     
PATIENCE_RESURRECT = 5  # Сколько эпох подряд должно быть 0 реанимаций для остановки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EMBEDDINGS = 2048   
EMBEDDING_DIM = 64      

def main():
    log_filename = f"train_4ch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout # Catch errors too

    print(f"Running on: {DEVICE}")
    print(f"Loading {DATA_FILE}...")
    
    try:
        data_tensor = torch.load(DATA_FILE)
    except FileNotFoundError:
        print("Файл не найден!")
        return

    print(f"Dataset shape: {data_tensor.shape}")

    dataset_size = len(data_tensor)
    val_size = min(100000, int(dataset_size * 0.1)) 
    train_size = dataset_size - val_size
    train_ds, val_ds = random_split(TensorDataset(data_tensor), [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = EEG_VQ_VAE(
        input_dim=INPUT_DIM, 
        hidden_dim=256,
        embedding_dim=EMBEDDING_DIM, 
        num_embeddings=NUM_EMBEDDINGS
    ).to(DEVICE)
    
    model.init_codebook_from_data(train_loader, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_recon = float('inf')
    zero_res_epochs = 0
    
    print(f"Начинаем автоматическое обучение до стабилизации словаря...")
    
    try:
        for epoch in range(MAX_EPOCHS):
            # Динамическое увеличение порога "смерти" для стабилизации
            # С каждой эпохой даем токенам больше времени на "выживание"
            model._vq_vae.dead_threshold = 100 + epoch * 100 
            
            model.train()
            train_recon_acc = 0
            train_vq_acc = 0
            epoch_resurrected = 0
            
            for batch in train_loader:
                x = batch[0].to(DEVICE)
                
                vq_loss, x_recon, n_res = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                loss = recon_loss + vq_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_recon_acc += recon_loss.item()
                train_vq_acc += vq_loss.item()
                epoch_resurrected += n_res
                
            # Валидация
            model.eval()
            val_recon_acc = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(DEVICE)
                    vq_loss, x_recon, _ = model(x)
                    val_recon_acc += F.mse_loss(x_recon, x).item()
            
            avg_val_recon = val_recon_acc / len(val_loader)
            n_tr = len(train_loader)
            
            print(f"Epoch {epoch+1:2d} | Recon: {train_recon_acc/n_tr:.6f} | VQ: {train_vq_acc/n_tr:.6f} | Val Recon: {avg_val_recon:.6f} | Res: {int(epoch_resurrected)} | Thr: {model._vq_vae.dead_threshold}")

            # Сохранение чекпоинта каждые 5 эпох
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), "vq_vae_4ch_checkpoint.weights")
                print(f"--- Чекпоинт сохранен (эпоха {epoch+1}) ---")

            # Сохранение лучшего результата (ТОЛЬКО если словарь стабилен)
            if int(epoch_resurrected) == 0:
                if avg_val_recon < best_val_recon:
                    best_val_recon = avg_val_recon
                    torch.save(model.state_dict(), "vq_vae_4ch_best.weights")
                    print(f"--- Новое лучшее стабильное состояние: {best_val_recon:.6f} ---")
                zero_res_epochs += 1
            else:
                zero_res_epochs = 0
                
            if zero_res_epochs >= PATIENCE_RESURRECT:
                print(f"\n>>> Стабилизация достигнута: {zero_res_epochs} эпох без реанимаций. Остановка.")
                break

    except KeyboardInterrupt:
        print("\n[ОСТАНОВКА] Обучение прервано пользователем. Сохраняем текущие веса...")
    
    # Финальное сохранение текущего состояния
    torch.save(model.state_dict(), "vq_vae_4ch_final.weights")
    print("\nОбучение завершено.")
    if best_val_recon != float('inf'):
        print(f"Лучший Val Recon: {best_val_recon:.6f}")
    
    # Визуализация для лучшей модели
    print("Генерация финальных графиков (best model)...")
    model.load_state_dict(torch.load("vq_vae_4ch_best.weights"))
    model.eval()
    x_val_batch = next(iter(val_loader))[0].to(DEVICE)
    with torch.no_grad():
        _, x_recon_batch, _ = model(x_val_batch)
    
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()
    indices = np.random.choice(len(x_val_batch), 30, replace=False)
    for i, idx in enumerate(indices):
        orig, recon = x_val_batch[idx].cpu().numpy(), x_recon_batch[idx].cpu().numpy()
        axes[i].plot(orig, label="Orig", color='blue')
        axes[i].plot(recon, label="Recon", color='red', linestyle='--')
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("results_4ch.png")
    print("Готово!")

if __name__ == "__main__":
    main()