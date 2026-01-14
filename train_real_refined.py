import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from vq_vae import VectorQuantizer # Импортируем только квантователь

# --- УЛУЧШЕННАЯ МОДЕЛЬ ---
class Deeper_EEG_VQ_VAE(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, embedding_dim=64, num_embeddings=4096):
        super(Deeper_EEG_VQ_VAE, self).__init__()
        
        # Более глубокий энкодер: 256 -> 256 -> 128 -> 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Добавили BatchNorm для стабильности
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.Tanh() 
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Симметричный декодер: 64 -> 128 -> 256 -> 256
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.unsqueeze(2) # Для совместимости с VQ
        vq_loss, quantized, perplexity = self._vq_vae(z)
        quantized = quantized.squeeze(2)
        x_recon = self.decoder(quantized)
        return vq_loss, x_recon, perplexity

# --- НАСТРОЙКИ ---
DATA_FILE = "eeg_dataset_fz.pt"
BATCH_SIZE = 256
EPOCHS = 100            # Увеличили
LEARNING_RATE = 5e-4    # Чуть уменьшили для стабильности при большом словаре
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EMBEDDINGS = 4096   # Увеличили словарь в 4 раза
EMBEDDING_DIM = 64      

def main():
    print(f"Running on: {DEVICE}")
    print(f"Config: Dict Size={NUM_EMBEDDINGS}, Epochs={EPOCHS}")
    
    # 1. Загрузка
    try:
        data_tensor = torch.load(DATA_FILE)
    except FileNotFoundError:
        print("Файл данных не найден.")
        return

    dataset_size = len(data_tensor)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_ds, val_ds = random_split(TensorDataset(data_tensor), [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Инициализация новой модели
    model = Deeper_EEG_VQ_VAE(
        input_dim=256, 
        hidden_dim=256, # Шире скрытый слой
        embedding_dim=EMBEDDING_DIM, 
        num_embeddings=NUM_EMBEDDINGS
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Обучение
    print("Начинаем обучение...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            vq_loss, x_recon, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                vq_loss, x_recon, _ = model(x)
                loss = F.mse_loss(x_recon, x) + vq_loss
                val_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
             print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss/len(train_loader):.6f} | Val: {val_loss/len(val_loader):.6f}")

    # Сохранение
    torch.save(model.state_dict(), "vq_vae_fz_refined.weights")
    
    # 4. Большая Визуализация (30 графиков)
    print("Генерируем 30 графиков...")
    model.eval()
    
    # Собираем данные из валидации
    x_val_batch = next(iter(val_loader))[0].to(DEVICE)
    with torch.no_grad():
        _, x_recon_batch, _ = model(x_val_batch)
    
    # Рисуем сетку 6x5
    fig, axes = plt.subplots(6, 5, figsize=(20, 18)) # Большой размер
    axes = axes.flatten()
    
    # Выбираем 30 случайных индексов
    indices = np.random.choice(len(x_val_batch), 30, replace=False)
    
    for i, idx in enumerate(indices):
        orig = x_val_batch[idx].cpu().numpy()
        recon = x_recon_batch[idx].cpu().numpy()
        
        ax = axes[i]
        ax.plot(orig, label="Orig", color='blue', linewidth=1)
        ax.plot(recon, label="Recon", color='red', linestyle='--', linewidth=1)
        ax.set_xticks([]) # Убираем подписи осей для чистоты
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("refined_results_30.png", dpi=150) # Высокое качество
    print("Готово! Смотри refined_results_30.png")

if __name__ == "__main__":
    main()