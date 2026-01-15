import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    # Добавили аргумент use_revival
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.02, use_revival=True):
        super(VectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self.use_revival = use_revival # Запоминаем флаг
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # inputs: [Batch, Channel, Length] -> [Batch, Length, Channel]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # L2 Norm
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        embedding_weight_norm = F.normalize(self._embedding.weight, p=2, dim=1)
        
        # Distances
        distances = (2 - 2 * torch.matmul(flat_input_norm, embedding_weight_norm.t()))
            
        # Indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # --- AGGRESSIVE DEAD CODE REVIVAL ---
        # Теперь проверяем флаг
        if self.training and self.use_revival:
            # Считаем использование каждого токена в батче
            # (bincount работает быстро на плоских индексах)
            counts = torch.bincount(encoding_indices.flatten(), minlength=self._num_embeddings)
            
            # Находим "ленивые" токены (использовались менее 2 раз в батче)
            # При батче 4096 это значит, что токен почти мертв
            dead_indices = torch.nonzero(counts < 2, as_tuple=False).flatten()
            
            if len(dead_indices) > 0:
                # Выбираем случайные входы, чтобы перезаписать мертвецов
                rand_inputs_idx = torch.randint(0, flat_input_norm.size(0), (len(dead_indices),), device=inputs.device)
                new_weights = flat_input_norm[rand_inputs_idx].detach()
                
                with torch.no_grad():
                    # Добавляем немного шума, чтобы они не слиплись
                    self._embedding.weight.data[dead_indices] = new_weights + torch.randn_like(new_weights) * 0.01

        # Quantize (используем обновленный словарь)
        # ВАЖНО: пересчитываем индексы для векторов, которые мы только что воскресили?
        # Нет, это дорого. В этом шаге используем старые индексы, 
        # а в следующей итерации "воскрешенные" векторы начнут притягивать к себе данные.
        
        quantized = self._embedding(encoding_indices).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return loss, quantized, encoding_indices

class EEG_VQ_VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, embedding_dim=64, num_embeddings=4096, use_revival=True):
        super(EEG_VQ_VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Широкий слой
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), # 256
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim), # 64
        )
        
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, use_revival=use_revival)
        
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
        z = z.unsqueeze(2)
        vq_loss, quantized, indices = self._vq_vae(z) 
        quantized = quantized.squeeze(2)
        x_recon = self.decoder(quantized)
        
        # Perplexity
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
            
            # K-Means++ style init (просто random sampling)
            indices = torch.randperm(z.shape[0])[:n_emb]
            self._vq_vae._embedding.weight.data.copy_(z[indices])
            print(f"Словарь инициализирован {len(indices)} векторами.")