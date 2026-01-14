import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from vq_vae import VectorQuantizer

# --- КОПИЯ ТВОЕЙ УЛУЧШЕННОЙ МОДЕЛИ (чтобы загрузить веса) ---
class Deeper_EEG_VQ_VAE(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, embedding_dim=64, num_embeddings=4096):
        super(Deeper_EEG_VQ_VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.Tanh() 
        )
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim)
        # Декодер нам сейчас не нужен, но для загрузки state_dict он должен быть описан
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode_to_tokens(self, x):
        """Возвращает только индексы токенов"""
        z = self.encoder(x)
        z = z.unsqueeze(2)
        # Нам нужно вытащить внутренности VectorQuantizer
        # В твоем коде vq_vae возвращает (loss, quantized, perplexity, encodings, encoding_indices)
        # Но давай сделаем проще: пересчитаем расстояния руками, так надежнее
        
        # Или, если ты не менял vq_vae.py, там внутри:
        # encoding_indices = torch.argmin(distances, dim=1)
        # Нам нужно получить доступ к этому.
        
        # Самый простой способ, не меняя vq_vae.py - вызвать форвард и забрать 3-й элемент?
        # Нет, твой vq_vae возвращает: loss, quantized, perplexity. Индексов нет.
        
        # ХАК: Давай вручную найдем индексы, используя веса словаря
        # z: [Batch, EmbDim, 1] -> [Batch, EmbDim]
        flat_input = z.squeeze(2)
        
        # Веса словаря
        embeddings = self._vq_vae._embedding.weight # [4096, 64]
        
        # Считаем расстояния (L2)
        # dist = (x-y)^2 = x^2 + y^2 - 2xy
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_input, embeddings.t()))
                    
        indices = torch.argmin(distances, dim=1)
        return indices

# --- НАСТРОЙКИ ---
WEIGHTS_FILE = "vq_vae_fz_refined.weights"
DATA_FILE = "eeg_dataset_fz.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Загружаем модель
    model = Deeper_EEG_VQ_VAE(num_embeddings=4096).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_FILE))
    model.eval()
    print("Модель загружена.")

    # 2. Загружаем данные
    data = torch.load(DATA_FILE)
    loader = DataLoader(TensorDataset(data), batch_size=1024, shuffle=False)
    
    all_tokens = []
    
    print("Токенизация...")
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            indices = model.encode_to_tokens(x)
            all_tokens.append(indices.cpu())
            
    # Объединяем
    full_sequence = torch.cat(all_tokens).numpy()
    print(f"Всего токенов: {len(full_sequence)}")
    print(f"Пример последовательности (первые 50): {full_sequence[:50]}")
    
    # 3. Анализ использования словаря
    unique_tokens = len(np.unique(full_sequence))
    print(f"Использовано уникальных токенов: {unique_tokens} из 4096")
    
    # Гистограмма
    plt.figure(figsize=(12, 6))
    plt.hist(full_sequence, bins=100, log=True)
    plt.title("Частота использования токенов (Log Scale)")
    plt.xlabel("Token ID")
    plt.ylabel("Count")
    plt.savefig("token_dist.png")
    print("Гистограмма сохранена в token_dist.png")
    
    # Сохраним как 'текст' для просмотра
    np.savetxt("eeg_as_text.txt", full_sequence, fmt="%d", newline=" ")
    print("Текст сохранен в eeg_as_text.txt")

if __name__ == "__main__":
    import numpy as np
    main()