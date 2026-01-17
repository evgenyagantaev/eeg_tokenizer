import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryFSQ(nn.Module):
    """
    Специализированный квантователь для L=2 (бинарная сетка).
    Гарантирует разделение в 0.
    """
    def __init__(self, num_dims):
        super().__init__()
        self.num_dims = num_dims
        # Веса для перевода битов в число: [1, 2, 4, 8, ...]
        basis = [2**i for i in range(num_dims)]
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))

    def forward(self, z):
        # z: [Batch, 16] 
        # Жесткое квантование: >0 -> 0.5, <=0 -> -0.5
        # Используем sign, ноSTE (Straight-Through Estimator)
        z_quantized = torch.where(z > 0, torch.tensor(0.5, device=z.device), torch.tensor(-0.5, device=z.device))
        
        # STE: градиент пролетает напрямую
        z_out = z + (z_quantized - z).detach()
        
        # Индексы: 0 или 1
        bit_indices = (z_quantized > 0).long()
        flat_indices = (bit_indices * self.basis).sum(dim=-1)
        
        return z_out, flat_indices

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class EEG_FSQ_VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, levels=[2]*16):
        super().__init__()
        self.latent_dim = len(levels)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, self.latent_dim)
        )
        
        # Нормализация перед квантованием гарантирует, что 0 - это центр
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        
        self.quantizer = BinaryFSQ(self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0) # Ортогональность помогает разделить измерения
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        z = self.latent_norm(z)
        
        if self.training:
            # Латентный шум + Дропаут на каналы (заставляем использовать все биты)
            # Иногда зануляем случайные биты, чтобы модель не полагалась на один
            if np.random.random() < 0.2:
                mask = (torch.rand_like(z) > 0.1).float()
                z = z * mask
            z = z + torch.randn_like(z) * 0.1
            
        quantized, indices = self.quantizer(z)
        
        x_recon = self.decoder(quantized)
        
        # Добавляем очень маленький штраф на корреляцию битов 
        # (чтобы они не делали одно и то же)
        decorr_loss = 0
        if self.training:
            z_corr = torch.matmul(z.t(), z) / z.shape[0]
            decorr_loss = 0.001 * (z_corr - torch.eye(self.latent_dim, device=z.device)).pow(2).mean()

        return decorr_loss, x_recon, indices

# --- VQ-VAE классы не трогаем ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self.register_buffer('_usage_counter', torch.zeros(num_embeddings))
        self.dead_threshold = 3000
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        embedding_weight_norm = F.normalize(self._embedding.weight, p=2, dim=1)
        distances = (2 - 2 * torch.matmul(flat_input_norm, embedding_weight_norm.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        n_resurrected = 0
        if self.training:
            counts = torch.bincount(encoding_indices.flatten(), minlength=self._num_embeddings)
            self._usage_counter += 1
            self._usage_counter[counts > 0] = 0
            dead_indices = torch.nonzero(self._usage_counter > self.dead_threshold, as_tuple=False).flatten()
            if len(dead_indices) > 0:
                num_to_revive = min(len(dead_indices), flat_input_norm.size(0), 10)
                min_dist, _ = distances.min(dim=1)
                _, top_k_idx = torch.topk(min_dist, k=num_to_revive)
                actual_dead_idx = dead_indices[:num_to_revive]
                new_weights = flat_input_norm[top_k_idx].detach()
                with torch.no_grad():
                    self._embedding.weight.data[actual_dead_idx] = new_weights + torch.randn_like(new_weights) * 0.001
                    self._usage_counter[actual_dead_idx] = 0
                n_resurrected = num_to_revive
        quantized = F.embedding(encoding_indices, embedding_weight_norm).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input_norm.view(input_shape))
        q_latent_loss = F.mse_loss(quantized, flat_input_norm.view(input_shape).detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = flat_input_norm.view(input_shape) + (quantized - flat_input_norm.view(input_shape)).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        return loss, quantized, encoding_indices, n_resurrected

class EEG_VQ_VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, embedding_dim=64, num_embeddings=4096):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, embedding_dim))
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=1)
        z = z.unsqueeze(2)
        vq_loss, quantized, indices, n_resurrected = self._vq_vae(z) 
        quantized = quantized.squeeze(2)
        x_recon = self.decoder(quantized)
        return vq_loss, x_recon, float(n_resurrected)
    def init_codebook_from_data(self, data_loader, device):
        self.eval()
        all_data = []
        for i, batch in enumerate(data_loader):
            if i > 40: break
            all_data.append(batch[0].to(device))
        data = torch.cat(all_data, dim=0)
        with torch.no_grad():
            z = self.encoder(data)
            z = F.normalize(z, p=2, dim=1)
            n_emb = self._vq_vae._num_embeddings
            indices = torch.randperm(z.shape[0])[:n_emb]
            centroids = z[indices].clone()
            for i in range(10):
                dot_product = torch.matmul(z, centroids.t())
                labels = torch.argmax(dot_product, dim=1)
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(n_emb, device=device)
                new_centroids.index_add_(0, labels, z)
                counts.index_add_(0, labels, torch.ones(z.shape[0], device=device))
                mask = counts > 0
                new_centroids[mask] /= counts[mask].unsqueeze(1)
                if (counts == 0).any():
                    num_empty = (counts == 0).sum().item()
                    new_indices = torch.randperm(z.shape[0])[:num_empty]
                    new_centroids[~mask] = z[new_indices]
                centroids = F.normalize(new_centroids, p=2, dim=1)
            self._vq_vae._embedding.weight.data.copy_(centroids)