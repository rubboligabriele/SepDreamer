import torch
import torch.nn as nn


class PerFeatureEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))

    def forward(self, x):
        # x: [B, D]
        # output: [B, D, k]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FactorizationMachine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, D, E]
        square_of_sum = torch.sum(x, dim=1) ** 2      # [B, E]
        sum_of_square = torch.sum(x ** 2, dim=1)      # [B, E]
        return 0.5 * (square_of_sum - sum_of_square)  # [B, E]


class AFIEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.outdim = 2 * embed_dim

        # embeddings for observations and deltas
        self.obs_embedding = PerFeatureEmbedding(input_dim, embed_dim)
        self.delta_embedding = PerFeatureEmbedding(input_dim, embed_dim)

        # produces a 2k linear term
        self.linear = nn.Linear(input_dim, 2 * embed_dim)

        self.fm = FactorizationMachine()

    def forward(self, x, delta):
        """
        x:     [B, D]
        delta: [B, D]
        """
        x_emb = self.obs_embedding(x)          # [B, D, k]
        d_emb = self.delta_embedding(delta)    # [B, D, k]

        embeddings = torch.cat([x_emb, d_emb], dim=-1)   # [B, D, 2k]

        linear_part = self.linear(x)           # [B, 2k]
        fm_part = self.fm(embeddings)          # [B, 2k]

        out = torch.sigmoid(linear_part + fm_part)   # [B, 2k]
        return out