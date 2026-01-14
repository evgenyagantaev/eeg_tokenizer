import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from vq_vae import VectorQuantizer

# --- ИСПРАВЛЕННЫЙ КЛАСС (С ДЕКОДЕРОМ) ---
class Final_EEG_VQ_VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, embedding_dim=64, num_embeddings=4096):
        super(Final_EEG_VQ_VAE, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim)
        
        # --- ВЕРНУЛИ ДЕКОДЕР (чтобы веса загрузились без ошибок) ---
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        # Прогоняем только через энкодер
        z = self.encoder(x)
        z = z.unsqueeze(2)
        
        # Ручной поиск ближайших токенов
        flat_input = z.squeeze(2)
        embeddings = self._vq_vae._embedding.weight
        
        # L2 расстояние
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_input, embeddings.t()))
        
        indices = torch.argmin(distances, dim=1)
        return indices

# --- НАСТРОЙКИ ---
DATA_ROOT = "/mnt/c/workspace/ibrain/NeoNeuroEngine/eeg-data"
WEIGHTS_FILE = "vq_vae_final.weights"
OUTPUT_TEXT_FILE = "eeg_corpus.txt"

WINDOW_SIZE = 512
STRIDE_INFERENCE = 128  # 1 сек = 2 токена
TARGET_SPECTRUM_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = Final_EEG_VQ_VAE().to(DEVICE)
    # Теперь структура совпадает, ошибки не будет
    model.load_state_dict(torch.load(WEIGHTS_FILE))
    model.eval()
    return model

def process_session(folder_path, model):
    file_path = os.path.join(folder_path, "game-raw-eeg-filtered.json")
    
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r') as f:
            data_json = json.load(f)
            
        if "recorded_data" not in data_json:
            return None
            
        section = data_json["recorded_data"]
        raw_data = np.array(section['data'])
        cols = section.get('cols', 5)
        
        # Решейп
        try:
            valid_len = (len(raw_data) // cols) * cols
            reshaped_data = raw_data[:valid_len].reshape(-1, cols)
            fz_signal = reshaped_data[:, 0]
        except:
            return None
        
        # --- ТОКЕНИЗАЦИЯ ---
        patches = []
        
        # Идем окном
        for i in range(0, len(fz_signal) - WINDOW_SIZE + 1, STRIDE_INFERENCE):
            patch = fz_signal[i : i + WINDOW_SIZE]
            
            # FFT
            complex_spectrum = np.fft.rfft(patch)
            magnitude_spectrum = np.abs(complex_spectrum)
            magnitude_spectrum = magnitude_spectrum[:TARGET_SPECTRUM_SIZE]
            
            patches.append(magnitude_spectrum)
            
        if not patches:
            return None
            
        patches_np = np.stack(patches)
        
        # Нормализация
        log_spectra = np.log1p(patches_np)
        max_val = np.max(log_spectra) + 1e-6
        norm_spectra = log_spectra / max_val
        
        tensor_data = torch.tensor(norm_spectra, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            tokens = model.encode(tensor_data)
            
        return tokens.cpu().numpy()

    except Exception as e:
        print(f"Error in {folder_path}: {e}")
        return None

def main():
    model = load_model()
    root_path = Path(DATA_ROOT)
    session_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    print(f"Обработка {len(session_folders)} сессий...")
    
    with open(OUTPUT_TEXT_FILE, 'w') as f_out:
        for folder in tqdm(session_folders):
            tokens = process_session(str(folder), model)
            
            if tokens is not None and len(tokens) > 0:
                tokens_str = " ".join(map(str, tokens))
                f_out.write(tokens_str + "\n")
                
    print(f"Готово! Корпус сохранен в {OUTPUT_TEXT_FILE}")

if __name__ == "__main__":
    main()