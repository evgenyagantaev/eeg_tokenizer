import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from vq_vae import EEG_VQ_VAE  # Импорт твоей модели

# --- НАСТРОЙКИ ---
DATA_FILE = "eeg_dataset_fz.pt"
BATCH_SIZE = 256        # Побольше, так как данных много
EPOCHS = 50             # Дадим ему время выучить паттерны
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Параметры словаря
NUM_EMBEDDINGS = 1024   # Размер словаря (количество "слов")
EMBEDDING_DIM = 64      # Размер вектора токена

def main():
    print(f"Running on: {DEVICE}")
    
    # 1. Загрузка данных
    try:
        data_tensor = torch.load(DATA_FILE)
        print(f"Загружен датасет: {data_tensor.shape}")
    except FileNotFoundError:
        print(f"Файл {DATA_FILE} не найден! Сначала запусти prepare_dataset.py")
        return

    # Разбиваем на Train / Validation (90% / 10%)
    # Чтобы проверить, что он не просто запомнил, а понял суть
    dataset_size = len(data_tensor)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_ds, val_ds = random_split(TensorDataset(data_tensor), [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Модель
    model = EEG_VQ_VAE(
        input_dim=256, 
        hidden_dim=128, 
        embedding_dim=EMBEDDING_DIM, 
        num_embeddings=NUM_EMBEDDINGS
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Обучение
    train_losses = []
    val_losses = []
    
    print("Начинаем обучение...")
    
    for epoch in range(EPOCHS):
        # --- Train Loop ---
        model.train()
        train_epoch_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE) # TensorDataset возвращает кортеж (x,)
            
            vq_loss, x_recon, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation Loop ---
        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                vq_loss, x_recon, _ = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                loss = recon_loss + vq_loss
                val_epoch_loss += loss.item()
                
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Вывод прогресса
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 4. Сохранение весов
    torch.save(model.state_dict(), "vq_vae_fz.weights")
    print("Веса модели сохранены в vq_vae_fz.weights")

    # 5. Визуализация (Берем 3 случайных примера из валидации)
    print("Рисуем результаты...")
    model.eval()
    
    # Берем один батч из валидации
    x_val_batch = next(iter(val_loader))[0].to(DEVICE)
    with torch.no_grad():
        _, x_recon_batch, _ = model(x_val_batch)
    
    # Рисуем 3 графика
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    indices = np.random.choice(len(x_val_batch), 3, replace=False)
    
    for i, idx in enumerate(indices):
        orig = x_val_batch[idx].cpu().numpy()
        recon = x_recon_batch[idx].cpu().numpy()
        
        axes[i].plot(orig, label="Real EEG Spectrum", color='blue', alpha=0.7)
        axes[i].plot(recon, label="Reconstructed via Tokens", color='red', linestyle='--', alpha=0.9)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("real_eeg_reconstruction.png")
    print("График сохранен в real_eeg_reconstruction.png")
    
    # Рисуем график Loss
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training Dynamics')
    plt.legend()
    plt.savefig("loss_curve.png")

if __name__ == "__main__":
    main()