import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1): # Снизил cost с 0.25 до 0.1
        super(VectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # Инициализация
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # inputs: [Batch, Channel, Length] -> [Batch, Length, Channel]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # --- L2 NORMALIZATION ---
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        embedding_weight_norm = F.normalize(self._embedding.weight, p=2, dim=1)
        
        # Расстояния
        distances = (2 - 2 * torch.matmul(flat_input_norm, embedding_weight_norm.t()))
            
        # Индексы
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # --- DEAD CODE REVIVAL (РЕАНИМАЦИЯ) ---
        # Работает только при обучении
        if self.training:
            # 1. Какие индексы были использованы в этом батче?
            unique_indices = encoding_indices.unique()
            
            # 2. Если использовано слишком мало токенов (меньше чем размер словаря)
            # То ищем "мертвые" индексы, которых нет в unique_indices
            # (Для скорости делаем простую эвристику: если батч большой (2048), а токенов мало)
            
            # Полный список всех индексов
            all_indices = torch.arange(self._num_embeddings, device=inputs.device)
            
            # Маска использованных (true если индекс был использован)
            # Это может быть медленно на каждом шаге, но с батчем 2048 это нормально
            used_mask = torch.zeros(self._num_embeddings, dtype=torch.bool, device=inputs.device)
            used_mask[unique_indices] = True
            
            # Список мертвых
            dead_indices = all_indices[~used_mask]
            
            if len(dead_indices) > 0:
                # Берем случайные векторы из ВХОДА (flat_input_norm), чтобы перезаписать мертвые
                # Выбираем N случайных индексов из батча
                rand_inputs_idx = torch.randint(0, flat_input_norm.size(0), (len(dead_indices),), device=inputs.device)
                
                # Берем сами векторы (важно: detach, чтобы не ломать градиенты энкодера)
                new_weights = flat_input_norm[rand_inputs_idx].detach()
                
                # Перезаписываем мертвые векторы в словаре
                # with torch.no_grad() обязательно!
                with torch.no_grad():
                    self._embedding.weight.data[dead_indices] = new_weights

        # Квантование (используем обновленный словарь)
        # Важно: берем веса заново, так как мы их могли обновить
        quantized = self._embedding(encoding_indices).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # STE
        quantized = inputs + (quantized - inputs).detach()
        
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return loss, quantized, encoding_indices

# --- МОДЕЛЬ (Без изменений) ---
class EEG_VQ_VAE(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, embedding_dim=64, num_embeddings=4096):
        super(EEG_VQ_VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.unsqueeze(2)
        vq_loss, quantized, indices = self._vq_vae(z) 
        quantized = quantized.squeeze(2)
        x_recon = self.decoder(quantized)
        
        # Perplexity для логов
        encodings = torch.zeros(indices.shape[0], self._vq_vae._num_embeddings, device=x.device)
        encodings.scatter_(1, indices, 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return vq_loss, x_recon, perplexity

    def init_codebook_from_data(self, data_loader, device):
        print("Инициализация словаря...")
        all_data = []
        for i, batch in enumerate(data_loader):
            if i > 20: break
            all_data.append(batch[0].to(device))
        
        data = torch.cat(all_data, dim=0)
        with torch.no_grad():
            z = self.encoder(data)
            z = F.normalize(z, p=2, dim=1)
            n_emb = self._vq_vae._num_embeddings
            indices = torch.randperm(z.shape[0])[:n_emb]
            self._vq_vae._embedding.weight.data.copy_(z[indices])
            print(f"Словарь инициализирован {len(indices)} векторами.")