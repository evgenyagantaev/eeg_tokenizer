import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        # --- NEW: Буферы для умной реанимации ---
        self.register_buffer('_usage_counter', torch.zeros(num_embeddings))
        self.dead_threshold = 3000
        self.max_resurrect_per_batch = 10

    def forward(self, inputs):
        # inputs: [Batch, Length, Channel] (после permute)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # L2 Normalization (Косинусное расстояние)
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        embedding_weight_norm = F.normalize(self._embedding.weight, p=2, dim=1)
        
        distances = (2 - 2 * torch.matmul(flat_input_norm, embedding_weight_norm.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # --- IMPROVED DEAD CODE REVIVAL ---
        n_resurrected = 0
        if self.training:
            # Считаем использование в текущем батче
            counts = torch.bincount(encoding_indices.flatten(), minlength=self._num_embeddings)
            
            # Обновляем счетчик игнора: +1 всем, сброс тем, кто использовался
            self._usage_counter += 1
            self._usage_counter[counts > 0] = 0
            
            # Ищем тех, кто "залежался"
            dead_indices = torch.nonzero(self._usage_counter > self.dead_threshold, as_tuple=False).flatten()
            
            if len(dead_indices) > 0:
                # Ограничиваем количество за один батч
                num_to_revive = min(len(dead_indices), flat_input_norm.size(0), self.max_resurrect_per_batch)
                
                # Находим "обиженные" входные векторы (максимальное расстояние до ближайшего кода)
                min_dist, _ = distances.min(dim=1)
                _, top_k_idx = torch.topk(min_dist, k=num_to_revive)
                
                # Реанимируем
                actual_dead_idx = dead_indices[:num_to_revive]
                new_weights = flat_input_norm[top_k_idx].detach()
                
                with torch.no_grad():
                    self._embedding.weight.data[actual_dead_idx] = new_weights + torch.randn_like(new_weights) * 0.001
                    self._usage_counter[actual_dead_idx] = 0
                
                n_resurrected = num_to_revive

        # Квантование (Используем нормализованные веса для консистентности)
        quantized = F.embedding(encoding_indices, embedding_weight_norm).view(input_shape)
        
        # Loss (теперь всё в едином масштабе [-1, 1])
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input_norm.view(input_shape))
        q_latent_loss = F.mse_loss(quantized, flat_input_norm.view(input_shape).detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # STE (Пробрасываем градиент к нормализованному входу)
        quantized = flat_input_norm.view(input_shape) + (quantized - flat_input_norm.view(input_shape)).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return loss, quantized, encoding_indices, n_resurrected

class EEG_VQ_VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, embedding_dim=64, num_embeddings=4096):
        super(EEG_VQ_VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), # 256
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim), # 64
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        # Важно: нормализуем выход энкодера перед VQ
        z = F.normalize(z, p=2, dim=1)
        
        z = z.unsqueeze(2)
        vq_loss, quantized, indices, n_resurrected = self._vq_vae(z) 
        quantized = quantized.squeeze(2)
        x_recon = self.decoder(quantized)
        
        return vq_loss, x_recon, float(n_resurrected)

    def init_codebook_from_data(self, data_loader, device):
        print("Инициализация словаря (K-Means)...")
        self.eval()
        all_data = []
        for i, batch in enumerate(data_loader):
            if i > 40: break # Собираем больше данных для качественной кластеризации
            all_data.append(batch[0].to(device))
        
        data = torch.cat(all_data, dim=0)
        with torch.no_grad():
            z = self.encoder(data)
            z = F.normalize(z, p=2, dim=1)
            
            n_emb = self._vq_vae._num_embeddings
            # Случайный выбор начальных центроидов из данных
            indices = torch.randperm(z.shape[0])[:n_emb]
            centroids = z[indices].clone()
            
            # 10 итераций мини-K-Means
            for i in range(10):
                # Находим ближайшие центроиды (через скалярное произведение)
                dot_product = torch.matmul(z, centroids.t())
                labels = torch.argmax(dot_product, dim=1)
                
                # Обновляем центроиды
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(n_emb, device=device)
                
                new_centroids.index_add_(0, labels, z)
                counts.index_add_(0, labels, torch.ones(z.shape[0], device=device))
                
                # Обработка пустых кластеров
                mask = counts > 0
                new_centroids[mask] /= counts[mask].unsqueeze(1)
                
                if (counts == 0).any():
                    num_empty = (counts == 0).sum().item()
                    new_indices = torch.randperm(z.shape[0])[:num_empty]
                    new_centroids[~mask] = z[new_indices]
                
                centroids = F.normalize(new_centroids, p=2, dim=1)
                print(f"  K-Means итерация {i+1}/10 завершена")

            self._vq_vae._embedding.weight.data.copy_(centroids)
            print(f"Словарь инициализирован {n_emb} центроидами K-Means.")