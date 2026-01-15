import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- НАСТРОЙКИ ---
# Укажи здесь имя своего последнего актуального файла
DATA_FILE = "eeg_dataset_fz_v2.pt" 
# Если у тебя уже есть 4-канальный файл, укажи его, скрипт поймет.

def main():
    print(f"Загрузка {DATA_FILE}...")
    try:
        data = torch.load(DATA_FILE)
    except FileNotFoundError:
        print("Файл не найден.")
        return

    print(f"Исходный размер: {data.shape}")

    # --- ПРЕПРОЦЕССИНГ ---
    # Нам нужно привести данные к виду [N_samples, 128_features]
    # Если у тебя [N, 512] (склеенные каналы), мы их разрежем, 
    # чтобы узнать сложность ИМЕННО СПЕКТРА, а не связей каналов.
    
    if data.dim() == 2:
        if data.shape[1] == 128:
            print("Формат: Одиночные спектры (128). Ок.")
            X = data.numpy()
        elif data.shape[1] == 512:
            print("Формат: Склеенные каналы (512). Разрезаем на 4 части по 128...")
            # [N, 512] -> [N, 4, 128] -> [N*4, 128]
            X = data.view(-1, 4, 128).view(-1, 128).numpy()
        elif data.shape[1] == 256:
             print("Формат: Старые спектры (256). Ок.")
             X = data.numpy()
        else:
            print(f"Непонятная размерность {data.shape[1]}, работаем как есть.")
            X = data.numpy()
    else:
        # Если вдруг [N, 4, 128]
        X = data.view(-1, data.shape[-1]).numpy()

    # Берем выборку, чтобы PCA не "повесил" комп (100к хватит с головой)
    MAX_SAMPLES = 100000
    if X.shape[0] > MAX_SAMPLES:
        print(f"Берем случайную выборку {MAX_SAMPLES} из {X.shape[0]}...")
        indices = np.random.choice(X.shape[0], MAX_SAMPLES, replace=False)
        X = X[indices]

    print(f"Данные для анализа: {X.shape}")

    # --- PCA ---
    print("Запуск PCA...")
    pca = PCA() # Считаем все компоненты
    pca.fit(X)

    # --- АНАЛИЗ ---
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Ищем точки пересечения порогов
    d_90 = np.argmax(cumsum >= 0.90) + 1
    d_95 = np.argmax(cumsum >= 0.95) + 1
    d_99 = np.argmax(cumsum >= 0.99) + 1
    
    print("-" * 30)
    print(f"РЕЗУЛЬТАТЫ АНАЛИЗА СЛОЖНОСТИ:")
    print(f"90% информации содержится в первых {d_90} компонентах")
    print(f"95% информации содержится в первых {d_95} компонентах")
    print(f"99% информации содержится в первых {d_99} компонентах")
    print("-" * 30)
    print(f"Твоя текущая размерность эмбеддинга: 64")
    if 64 >= d_99:
        print("Вердикт: 64 — ЭТО С ЗАПАСОМ. Можно сжимать сильнее.")
    elif 64 >= d_95:
        print("Вердикт: 64 — ЭТО ОТЛИЧНО. Баланс соблюден.")
    else:
        print("Вердикт: 64 — МАЛОВАТО. Теряем информацию.")

    # --- ВИЗУАЛИЗАЦИЯ ---
    plt.figure(figsize=(15, 10))

    # График 1: Накопленная дисперсия (Elbow plot)
    plt.subplot(2, 1, 1)
    plt.plot(cumsum, linewidth=2)
    plt.axvline(d_95, color='r', linestyle='--', label=f'95% Explained ({d_95} dims)')
    plt.axhline(0.95, color='r', linestyle='--', alpha=0.3)
    plt.axvline(64, color='g', linestyle='-', label='Current Embedding (64)')
    plt.xlabel('Количество компонент')
    plt.ylabel('Сохраненная информация (Variance)')
    plt.title('Сколько размерностей нужно твоему сигналу?')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 2: Топ-3 Главные Компоненты (Базовые формы)
    plt.subplot(2, 1, 2)
    plt.plot(pca.components_[0], label='1st Component (Главный тренд)')
    plt.plot(pca.components_[1], label='2nd Component')
    plt.plot(pca.components_[2], label='3rd Component')
    plt.title('Как выглядят главные "кирпичики" твоих данных (Eigen-Spectra)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_analysis.png")
    print("График сохранен в pca_analysis.png")

if __name__ == "__main__":
    main()