import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from vq_vae import EEG_VQ_VAE 

# --- НАСТРОЙКИ 4 КАНАЛА ---
DATA_FILE = "eeg_dataset_4ch_shared.pt"
INPUT_DIM = 128         # Спектр одного канала
BATCH_SIZE = 4096       # Увеличили для скорости
EPOCHS = 30             
LEARNING_RATE = 3e-4    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EMBEDDINGS = 4096   
EMBEDDING_DIM = 64      

def main():
    print(f"Running on: {DEVICE}")
    print(f"Loading {DATA_FILE}...")
    
    try:
        data_tensor = torch.load(DATA_FILE)
    except FileNotFoundError:
        print("Файл не найден!")
        return

    print(f"Dataset shape: {data_tensor.shape}")

    # Сплит (берем совсем чуть-чуть на валидацию, 100к хватит)
    dataset_size = len(data_tensor)
    val_size = 100000 
    train_size = dataset_size - val_size
    
    train_ds, val_ds = random_split(TensorDataset(data_tensor), [train_size, val_size])
    
    # num_workers=0 для стабильности в WSL
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Модель (та же самая, универсальная)
    model = EEG_VQ_VAE(
        input_dim=INPUT_DIM, 
        hidden_dim=512,
        embedding_dim=EMBEDDING_DIM, 
        num_embeddings=NUM_EMBEDDINGS
    ).to(DEVICE)
    
    # Инициализация словаря
    model.init_codebook_from_data(train_loader, DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Начинаем обучение на {len(train_loader)} батчах...")
    
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
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    # Сохранение
    torch.save(model.state_dict(), "vq_vae_4ch.weights")
    print("Веса сохранены в vq_vae_4ch.weights")
    
    # Визуализация
    print("Генерация графиков...")
    model.eval()
    x_val_batch = next(iter(val_loader))[0].to(DEVICE)
    with torch.no_grad():
        _, x_recon_batch, _ = model(x_val_batch)
    
    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    axes = axes.flatten()
    indices = np.random.choice(len(x_val_batch), 30, replace=False)
    
    for i, idx in enumerate(indices):
        orig = x_val_batch[idx].cpu().numpy()
        recon = x_recon_batch[idx].cpu().numpy()
        
        ax = axes[i]
        ax.plot(orig, label="Orig", color='blue')
        ax.plot(recon, label="Recon", color='red', linestyle='--')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("results_4ch.png")
    print("Готово!")

if __name__ == "__main__":
    main()