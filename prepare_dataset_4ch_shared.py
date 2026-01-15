import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import gc

# --- НАСТРОЙКИ ---
DATA_ROOT = "/mnt/c/workspace/ibrain/NeoNeuroEngine/eeg-data"
WINDOW_SIZE = 512
STRIDE = 15                 # Плотный шаг
TARGET_SPECTRUM_SIZE = 128  # 0-62.5 Гц
OUTPUT_FILE = "eeg_dataset_4ch_shared.pt"

# Параметры фильтрации
MIN_STD_DEV = 0.1
MAX_AMPLITUDE = 100.0 

# Индексы каналов в массиве [fz, c3, cz, c4, pz]
# Нам нужны: Fz(0), C3(1), C4(3), Pz(4)
CH_INDICES = [0, 1, 3, 4]

def is_clean_patch(signal_patch):
    """Проверка одного патча одного канала"""
    if np.std(signal_patch) < MIN_STD_DEV:
        return False
    if np.max(np.abs(signal_patch)) > MAX_AMPLITUDE:
        return False
    return True

def process_session_folder(folder_path):
    extracted_spectra = []
    
    # Ищем файлы с игровыми данными
    files_to_check = {
        "game-raw-eeg-filtered.json": ["recorded_data"]
    }
    
    for filename, keys in files_to_check.items():
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path): continue
            
        try:
            with open(file_path, 'r') as f:
                data_json = json.load(f)
                
            for key in keys:
                if key not in data_json: continue
                
                section = data_json[key]
                raw_data = np.array(section['data'])
                cols = section.get('cols', 5)
                
                # Решейп в [Samples, 5]
                try:
                    reshaped_data = raw_data.reshape(-1, cols)
                except:
                    valid_len = (len(raw_data) // cols) * cols
                    reshaped_data = raw_data[:valid_len].reshape(-1, cols)
                
                # --- ВЫБОР 4 КАНАЛОВ ---
                # Получаем [Samples, 4]
                signals_4ch = reshaped_data[:, CH_INDICES]
                
                num_samples = signals_4ch.shape[0]
                if num_samples < WINDOW_SIZE: continue

                # Проходим окном
                for i in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
                    # Вырезаем окно для всех 4 каналов сразу: [512, 4]
                    window_4ch = signals_4ch[i : i + WINDOW_SIZE, :]
                    
                    # Обрабатываем каждый канал НЕЗАВИСИМО
                    for ch in range(4):
                        single_signal = window_4ch[:, ch]
                        
                        # Фильтруем каждый канал отдельно
                        # (если на C3 шум, это не повод выкидывать чистый Fz)
                        if not is_clean_patch(single_signal):
                            continue
                        
                        # FFT
                        # rfft для одного канала
                        complex_spectrum = np.fft.rfft(single_signal)
                        magnitude_spectrum = np.abs(complex_spectrum)
                        
                        # Обрезаем
                        magnitude_spectrum = magnitude_spectrum[:TARGET_SPECTRUM_SIZE]
                        
                        extracted_spectra.append(magnitude_spectrum)
                
        except Exception as e:
            # print(f"Error: {e}")
            pass
            
    if extracted_spectra:
        # Возвращаем сразу float32 для экономии памяти
        return np.stack(extracted_spectra).astype(np.float32)
    else:
        return None

def main():
    all_spectra_chunks = []
    root_path = Path(DATA_ROOT)
    session_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    print(f"Обработка {len(session_folders)} сессий (4 канала, независимые патчи)...")
    
    total_count = 0
    
    for folder in tqdm(session_folders):
        chunk = process_session_folder(str(folder))
        if chunk is not None:
            all_spectra_chunks.append(chunk)
            total_count += len(chunk)
            
    if not all_spectra_chunks:
        print("Данные не найдены.")
        return

    print(f"Сборка общего массива ({total_count} спектров)...")
    
    # Собираем огромный массив
    # Внимание: это может занять 4-5 ГБ RAM
    full_dataset = np.concatenate(all_spectra_chunks, axis=0)
    
    # Чистим память
    del all_spectra_chunks
    gc.collect()
    
    print("Log-transform и нормализация...")
    # Log1p in-place
    np.log1p(full_dataset, out=full_dataset)
    
    # Max normalization
    max_val = np.max(full_dataset)
    print(f"Max value for normalization: {max_val}")
    full_dataset /= max_val
    
    print("Конвертация в Torch...")
    tensor_data = torch.from_numpy(full_dataset)
    
    print(f"Итоговый размер: {tensor_data.shape}")
    print(f"Сохранение в {OUTPUT_FILE}...")
    torch.save(tensor_data, OUTPUT_FILE)
    print("Готово!")

if __name__ == "__main__":
    main()