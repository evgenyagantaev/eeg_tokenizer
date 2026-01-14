import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

FILE_PATH = "eeg_corpus.txt"
VOCAB_SIZE = 4096

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Файл {FILE_PATH} не найден! Сначала запусти tokenize_to_text.py")
        return

    print(f"Чтение файла {FILE_PATH}...")
    
    all_tokens = []
    line_count = 0
    
    with open(FILE_PATH, 'r') as f:
        for line in f:
            # Читаем строку, разбиваем по пробелам, превращаем в int
            tokens = [int(t) for t in line.strip().split()]
            all_tokens.extend(tokens)
            line_count += 1

    if not all_tokens:
        print("Файл пуст!")
        return

    total_tokens = len(all_tokens)
    unique_tokens_set = set(all_tokens)
    unique_count = len(unique_tokens_set)
    
    print("-" * 30)
    print(f"Количество сессий (строк): {line_count}")
    print(f"Всего токенов в корпусе: {total_tokens}")
    print(f"Средняя длина сессии: {total_tokens / line_count:.1f} токенов")
    print("-" * 30)
    print(f"Использовано уникальных токенов: {unique_count} из {VOCAB_SIZE}")
    print(f"Заполнение словаря: {(unique_count / VOCAB_SIZE) * 100:.2f}%")
    print("-" * 30)

    # Топ самых частых
    counts = Counter(all_tokens)
    print("Топ-5 самых частых токенов:")
    for token, count in counts.most_common(5):
        print(f"Token ID {token}: {count} раз ({(count/total_tokens)*100:.2f}%)")

    # Визуализация
    print("Генерация гистограммы...")
    plt.figure(figsize=(15, 6))
    
    # Рисуем гистограмму по всем 4096 ID
    plt.hist(all_tokens, bins=200, color='royalblue', alpha=0.8)
    
    plt.title(f"Распределение токенов в корпусе (Unique: {unique_count}/{VOCAB_SIZE})")
    plt.xlabel("Token ID (0-4095)")
    plt.ylabel("Count (Log Scale)")
    plt.yscale('log') # Логарифмическая шкала важна, чтобы увидеть редкие токены
    plt.grid(True, alpha=0.3)
    
    output_img = "corpus_histogram.png"
    plt.savefig(output_img)
    print(f"График сохранен в {output_img}")

if __name__ == "__main__":
    main()