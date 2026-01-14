import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# --- НАСТРОЙКИ ---
# Путь к данным (адаптирован для WSL, если запускаешь из Windows Python, поменяй обратно на C:/...)
DATA_ROOT = "/mnt/c/workspace/ibrain/NeoNeuroEngine/eeg-data" 
WINDOW_SIZE = 512
TARGET_SPECTRUM_SIZE = 256
OUTPUT_FILE = "eeg_dataset_fz.pt"

def process_session_folder(folder_path):
    """Обрабатывает одну папку сессии, ищет нужные файлы."""
    extracted_patches = []
    
    # Список файлов и ключей, которые мы ищем внутри
    # baseline: bl_closed, bl_opened
    # game: recorded_data
    
    files_to_check = {
        "baseline-raw-eeg-filtered.json": ["bl_closed", "bl_opened"],
        "game-raw-eeg-filtered.json": ["recorded_data"]
    }
    
    for filename, keys in files_to_check.items():
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                data_json = json.load(f)
                
            for key in keys:
                if key not in data_json:
                    continue
                
                section = data_json[key]
                raw_data = np.array(section['data'])
                cols = section.get('cols', 5) # Обычно 5 каналов
                
                # --- ВЫДЕЛЕНИЕ КАНАЛА Fz ---
                # Данные лежат плоско: [fz, c3, cz, c4, pz, fz, c3, ...]
                # Нам нужен каждый 5-й элемент, начиная с 0 (так как Fz первый)
                
                # Решейпим в матрицу (N_samples, 5_channels)
                # -1 означает "посчитать количество строк автоматически"
                try:
                    reshaped_data = raw_data.reshape(-1, cols)
                except ValueError:
                    # Если данных не кратное количество, отбросим хвост
                    valid_len = (len(raw_data) // cols) * cols
                    reshaped_data = raw_data[:valid_len].reshape(-1, cols)
                
                # Берем 0-й столбец (Fz)
                fz_signal = reshaped_data[:, 0]
                
                # --- НАРЕЗКА НА ПАТЧИ (512) ---
                n_patches = len(fz_signal) // WINDOW_SIZE
                if n_patches == 0:
                    continue
                    
                # Обрезаем до целого числа патчей
                fz_signal = fz_signal[:n_patches * WINDOW_SIZE]
                
                # Превращаем в массив патчей [N, 512]
                patches = fz_signal.reshape(n_patches, WINDOW_SIZE)
                
                # --- FFT (Спектр) ---
                # rfft возвращает (512/2 + 1) = 257 точек комплексных
                complex_spectrum = np.fft.rfft(patches, axis=1)
                
                # Берем модуль (амплитуду)
                magnitude_spectrum = np.abs(complex_spectrum)
                
                # Отрезаем лишнее, чтобы осталось ровно 256 точек
                # (обычно 0-я точка - это DC offset, а 256-я - Найквист, 
                # для простоты модели берем первые 256)
                magnitude_spectrum = magnitude_spectrum[:, :TARGET_SPECTRUM_SIZE]
                
                extracted_patches.append(magnitude_spectrum)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if extracted_patches:
        return np.concatenate(extracted_patches, axis=0)
    else:
        return None

def main():
    all_spectra = []
    
    # Ищем все подпапки
    root_path = Path(DATA_ROOT)
    if not root_path.exists():
        print(f"Путь {DATA_ROOT} не найден! Проверь монтирование дисков в WSL.")
        return

    # Получаем список папок сессий
    session_folders = [f for f in root_path.iterdir() if f.is_dir()]
    print(f"Найдено сессий: {len(session_folders)}")
    
    for folder in tqdm(session_folders, desc="Processing sessions"):
        session_data = process_session_folder(str(folder))
        if session_data is not None:
            all_spectra.append(session_data)
            
    if not all_spectra:
        print("Не удалось извлечь данные.")
        return
        
    # Объединяем все в один огромный тензор
    full_dataset_numpy = np.concatenate(all_spectra, axis=0)
    
    # --- НОРМАЛИЗАЦИЯ ---
    # Важный момент: у спектров огромный разброс.
    # Для VQ-VAE лучше всего работает Log-transform, чтобы сжать динамический диапазон,
    # а затем MinMax или Standard scaling.
    
    print("Применяем Log-transform и нормализацию...")
    # Добавляем epsilon, чтобы не брать log(0)
    log_spectra = np.log1p(full_dataset_numpy) 
    
    # Считаем глобальный максимум для нормализации в [0, 1]
    # (Можно делать и по-батчево, но глобальная сохраняет относительную громкость)
    max_val = np.max(log_spectra)
    normalized_spectra = log_spectra / max_val
    
    # Конвертируем в Torch Tensor
    tensor_data = torch.tensor(normalized_spectra, dtype=torch.float32)
    
    print(f"Итоговый размер датасета: {tensor_data.shape}")
    print(f"Сохраняем в {OUTPUT_FILE}...")
    torch.save(tensor_data, OUTPUT_FILE)
    print("Готово!")

if __name__ == "__main__":
    main()