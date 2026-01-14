import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from vq_vae import EEG_VQ_VAE  # Импортируем твой класс из прошлого файла

# --- 1. Генератор синтетических данных (Имитация ЭЭГ) ---
def generate_synthetic_data(num_samples=10000, seq_len=512):
    # Создаем время
    t = np.linspace(0, 1, seq_len)
    data = []
    
    for _ in range(num_samples):
        # Случайная амплитуда и фаза для "Альфа" (10 Гц) и "Бета" (20 Гц)
        a1, a2 = np.random.uniform(0.1, 1.0), np.random.uniform(0.1, 0.5)
        p1, p2 = np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)
        
        # Сигнал = сумма синусоид + немного шума
        signal = a1 * np.sin(2 * np.pi * 10 * t + p1) + \
                 a2 * np.sin(2 * np.pi * 20 * t + p2) + \
                 np.random.normal(0, 0.1, seq_len)
        
        # Делаем Фурье (FFT) -> берем амплитуды (спектр)
        # rfft возвращает (seq_len/2 + 1) точек
        spectrum = np.abs(np.fft.rfft(signal))
        
        # Нормализуем (важно для нейросети!)
        spectrum = spectrum / np.max(spectrum)
        
        # Отрезаем лишнее, чтобы было ровно 256 точек (для удобства)
        data.append(spectrum[:256])
        
    return torch.tensor(np.array(data), dtype=torch.float32)

# --- 2. Настройки ---
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Подготовка ---
print("Генерация данных...")
full_dataset = generate_synthetic_data()
train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Данные готовы. Размер: {full_dataset.shape}")

# Создаем модель (256 входов -> сжатие в 64 -> 1024 токена в словаре)
model = EEG_VQ_VAE(input_dim=256, hidden_dim=128, embedding_dim=64, num_embeddings=1024).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Цикл обучения ---
print("Начинаем обучение...")
losses = []

for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        
        # Forward
        vq_loss, recon_batch, _ = model(batch)
        
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon_batch, batch)
        
        # Total Loss
        loss = recon_loss + vq_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

# --- 5. Визуализация результата ---
print("Обучение завершено. Рисуем результат...")

# Берем один пример из последних
model.eval()
with torch.no_grad():
    sample = full_dataset[0].unsqueeze(0).to(DEVICE) # Берем первый пример
    _, reconstructed, _ = model(sample)

# Переводим на CPU для рисования
orig = sample.cpu().numpy()[0]
recon = reconstructed.cpu().numpy()[0]

plt.figure(figsize=(10, 5))
plt.plot(orig, label="Оригинал (Спектр)", color='blue', alpha=0.7)
plt.plot(recon, label="Восстановлено из Токена", color='red', linestyle='--', alpha=0.9)
plt.title("Работа VQ-VAE: Сжатие и восстановление спектра")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("result.png") # Сохраним в файл
print("График сохранен в result.png")