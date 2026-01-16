import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from vq_vae import EEG_VQ_VAE 
import sys
import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

DATA_FILE = "eeg_dataset_4ch_shared.pt"
WEIGHTS_FILE = "vq_vae_4ch.weights" # Грузим текущие
NEW_WEIGHTS_FILE = "vq_vae_4ch_stable.weights" # Сохраним новые

INPUT_DIM = 128
BATCH_SIZE = 4096
EPOCHS = 10             
LEARNING_RATE = 5e-5    # <--- ОЧЕНЬ МАЛЕНЬКИЙ (в 6 раз меньше)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    log_filename = f"train_stabilize_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout # Catch errors too

    print(f"Running on: {DEVICE}")
    data_tensor = torch.load(DATA_FILE)
    
    # Dataloaders
    dataset_size = len(data_tensor)
    train_size = int(0.95 * dataset_size)
    val_size = dataset_size - train_size
    train_ds, val_ds = random_split(TensorDataset(data_tensor), [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # --- МОДЕЛЬ (ОТКЛЮЧАЕМ REVIVAL) ---
    model = EEG_VQ_VAE(
        input_dim=INPUT_DIM, 
        hidden_dim=512,
        embedding_dim=64, 
        num_embeddings=4096,
        use_revival=False # <--- ВАЖНО: Выключаем реанимацию
    ).to(DEVICE)
    
    # Загружаем старые веса
    model.load_state_dict(torch.load(WEIGHTS_FILE))
    print("Веса загружены. Начинаем стабилизацию...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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

    torch.save(model.state_dict(), NEW_WEIGHTS_FILE)
    print(f"Стабильные веса сохранены в {NEW_WEIGHTS_FILE}")

if __name__ == "__main__":
    main()