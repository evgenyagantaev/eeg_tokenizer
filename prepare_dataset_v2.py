import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import gc


# --- НОВЫЕ НАСТРОЙКИ ---
DATA_ROOT = "/mnt/c/workspace/ibrain/NeoNeuroEngine/eeg-data"
WINDOW_SIZE = 512
STRIDE = 15                 # Шаг окна (Overlap)
TARGET_SPECTRUM_SIZE = 128  # Обрезаем спектр (0 - 62.5 Гц)

# Критерии отбраковки (нужно подбирать экспериментально, но начнем с эвристик)
# Данные в JSON, скорее всего, в микровольтах (uV) или вольтах.
# Судя по твоим примерам (-1.6, 1.5), это похоже на uV после фильтрации или Z-score.
# Давай считать, что это uV.
MIN_STD_DEV = 0.1     # Если сигнал плоский (почти ноль) -> мусор
MAX_AMPLITUDE = 100.0 # Если есть пик выше 100 (артефакт движения) -> мусор

OUTPUT_FILE = "eeg_dataset_fz_v2.pt"

def is_clean_patch(signal_patch):
    """Проверяет патч на наличие артефактов."""
    # 1. Проверка на "мертвый" сигнал
    if np.std(signal_patch) < MIN_STD_DEV:
        return False
    
    # 2. Проверка на "взрывы" (высокоамплитудные артефакты)
    # Используем абсолютное значение, так как сигнал колеблется около нуля
    if np.max(np.abs(signal_patch)) > MAX_AMPLITUDE:
        return False
        
    return True

def process_session_folder(folder_path):
    extracted_patches = []
    
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
                cols = section.get('cols', 5)
                
                # Решейп и выделение Fz
                try:
                    reshaped_data = raw_data.reshape(-1, cols)
                except ValueError:
                    valid_len = (len(raw_data) // cols) * cols
                    reshaped_data = raw_data[:valid_len].reshape(-1, cols)
                
                fz_signal = reshaped_data[:, 0]
                
                # --- СЛАЙДИНГ (Скользящее окно) ---
                # Теперь мы идем не шагами по 512, а по STRIDE
                num_samples = len(fz_signal)
                
                # Если сигнал короче окна, пропускаем
                if num_samples < WINDOW_SIZE:
                    continue

                for i in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
                    patch = fz_signal[i : i + WINDOW_SIZE]
                    
                    # --- ФИЛЬТРАЦИЯ "ГРЯЗИ" ---
                    if not is_clean_patch(patch):
                        continue
                    
                    # --- FFT ---
                    complex_spectrum = np.fft.rfft(patch)
                    magnitude_spectrum = np.abs(complex_spectrum)
                    
                    # --- ОБРЕЗАНИЕ СПЕКТРА ---
                    # Берем только первые 128 частот (0 - 62.5 Гц)
                    magnitude_spectrum = magnitude_spectrum[:TARGET_SPECTRUM_SIZE]
                    
                    extracted_patches.append(magnitude_spectrum)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if extracted_patches:
        return np.stack(extracted_patches) # stack быстрее concatenate для списка массивов
    else:
        return None

def main():
    all_spectra = []
    root_path = Path(DATA_ROOT)
    session_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    print(f"Обработка {len(session_folders)} сессий...")
    
    total_patches = 0
    
    # 1. Сбор данных
    for folder in tqdm(session_folders):
        session_data = process_session_folder(str(folder))
        if session_data is not None:
            # Сразу конвертируем в float32 для экономии памяти (в 2 раза меньше)
            all_spectra.append(session_data.astype(np.float32))
            total_patches += len(session_data)
            
    if not all_spectra:
        print("Данные не найдены.")
        return

    print(f"Сборка общего тензора ({total_patches} патчей)...")
    
    # Собираем сразу в float32
    full_dataset_numpy = np.concatenate(all_spectra, axis=0, dtype=np.float32)
    
    # ВАЖНО: Сразу удаляем список и чистим мусор, чтобы освободить 2.5 ГБ
    del all_spectra
    gc.collect()
    
    print("Log-transform (in-place)...")
    # Используем параметр out=, чтобы менять массив на месте, не создавая копию
    np.log1p(full_dataset_numpy, out=full_dataset_numpy)
    
    print("Нормализация (in-place)...")
    max_val = np.max(full_dataset_numpy)
    
    # Деление на месте
    full_dataset_numpy /= max_val
    
    print("Конвертация в тензор (zero-copy)...")
    # torch.from_numpy не копирует память, а использует тот же массив!
    tensor_data = torch.from_numpy(full_dataset_numpy)
    
    print(f"Итоговый размер: {tensor_data.shape}")
    print(f"Сохранение в {OUTPUT_FILE}...")
    torch.save(tensor_data, OUTPUT_FILE)
    print("Готово!")

if __name__ == "__main__":
    main()