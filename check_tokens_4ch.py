import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from vq_vae import EEG_VQ_VAE  # Импортируем твою модель

# --- НАСТРОЙКИ ---
DATA_ROOT = "/mnt/c/workspace/ibrain/NeoNeuroEngine/eeg-data"
WEIGHTS_FILE = "vq_vae_4ch.weights"
CH_INDICES = [0, 1, 3, 4]  # Fz, C3, C4, Pz

# Параметры инференса (как для LLM)
WINDOW_SIZE = 512
STRIDE = 128            # Шаг 0.5 секунды
TARGET_SPECTRUM_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    # Модель была обучена на векторе 128
    model = EEG_VQ_VAE(
        input_dim=128, 
        hidden_dim=512,
        embedding_dim=64, 
        num_embeddings=4096
    ).to(DEVICE)
    
    # Загружаем веса
    # map_location='cuda' важно, если сохраняли на gpu
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
    model.eval()
    return model

def get_tokens_from_session(folder_path, model):
    file_path = os.path.join(folder_path, "game-raw-eeg-filtered.json")
    if not os.path.exists(file_path): return []
    
    try:
        with open(file_path, 'r') as f:
            data_json = json.load(f)
            
        if "recorded_data" not in data_json: return []
        
        section = data_json["recorded_data"]
        raw_data = np.array(section['data'])
        cols = section.get('cols', 5)
        
        # Решейп
        try:
            valid_len = (len(raw_data) // cols) * cols
            reshaped_data = raw_data[:valid_len].reshape(-1, cols)
        except:
            return []
            
        # Выбираем 4 канала: [Samples, 4]
        signals = reshaped_data[:, CH_INDICES]
        num_samples = signals.shape[0]
        
        if num_samples < WINDOW_SIZE: return []

        batch_spectra = []
        
        # Скользим окном
        for i in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
            # [512, 4]
            window = signals[i : i + WINDOW_SIZE, :]
            
            # FFT для каждого канала (axis=0, т.к. время - это строки)
            # window.T -> [4, 512] для удобства fft
            complex_spectrum = np.fft.rfft(window.T, axis=1)
            magnitude = np.abs(complex_spectrum)
            
            # Обрезаем до 128
            magnitude = magnitude[:, :TARGET_SPECTRUM_SIZE] # [4, 128]
            
            # Добавляем в общий список. 
            # Мы хотим просто собрать статистику токенов, поэтому
            # кладем их как отдельные примеры
            batch_spectra.append(magnitude)
            
        if not batch_spectra: return []
        
        # [N_windows, 4, 128] -> [N_windows * 4, 128]
        # Мы сваливаем все каналы в одну кучу, так как словарь у нас общий
        batch_np = np.concatenate(batch_spectra, axis=0)
        
        # --- Нормализация (Log1p + Max по сессии) ---
        log_spectra = np.log1p(batch_np)
        max_val = np.max(log_spectra) + 1e-6
        norm_spectra = log_spectra / max_val
        
        # Инференс
        tensor_data = torch.from_numpy(norm_spectra).float().to(DEVICE)
        
        with torch.no_grad():
            # Нам нужно вытащить индексы.
            # В классе EEG_VQ_VAE (если ты не менял return) 
            # forward возвращает: vq_loss, x_recon, perplexity
            # Нам придется использовать энкодер + vq напрямую
            
            z = model.encoder(tensor_data)
            z = z.unsqueeze(2)
            # vq_vae возвращает (loss, quantized, indices) - если ты обновил класс как мы договаривались
            # Если нет, то используем метод encode_to_tokens или хак
            
            # Используем хак для надежности (если класс старый)
            # Или лучше: используем forward VQ, если он возвращает индексы
            _, _, indices = model._vq_vae(z)
            
        return indices.view(-1).cpu().tolist()

    except Exception as e:
        # print(f"Err: {e}")
        return []

def main():
    model = load_model()
    root_path = Path(DATA_ROOT)
    session_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    print(f"Анализ {len(session_folders)} сессий...")
    
    all_tokens = []
    
    for folder in tqdm(session_folders):
        tokens = get_tokens_from_session(str(folder), model)
        all_tokens.extend(tokens)
        
    if not all_tokens:
        print("Токенов нет.")
        return

    # --- СТАТИСТИКА ---
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    vocab_size = 4096
    
    print("=" * 40)
    print(f"Всего токенов: {total_tokens}")
    print(f"Использовано уникальных: {unique_tokens} из {vocab_size}")
    print(f"Заполнение: {(unique_tokens/vocab_size)*100:.2f}%")
    print("=" * 40)
    
    # Топ частых
    counts = Counter(all_tokens)
    print("Топ-5 частых:")
    for t, c in counts.most_common(5):
        print(f"ID {t}: {c} ({c/total_tokens:.2%})")

    # Гистограмма
    plt.figure(figsize=(15, 6))
    plt.hist(all_tokens, bins=vocab_size//4, color='green', alpha=0.7) # bins поменьше для читаемости
    plt.yscale('log')
    plt.title(f"Распределение 4-канальных токенов (Unique: {unique_tokens})")
    plt.xlabel("Token ID")
    plt.ylabel("Count (Log)")
    plt.savefig("tokens_4ch_dist.png")
    print("График: tokens_4ch_dist.png")

if __name__ == "__main__":
    main()