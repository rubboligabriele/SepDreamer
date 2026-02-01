import torch
import torch.nn as nn
    
class FeaturesLinear(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_features)``
        """
        return self.linear(x)  # Linear term for each feature


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_features, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2  # Sum features and square the result
        sum_of_square = torch.sum(x ** 2, dim=1)  # Square each feature and sum
        ix = square_of_sum - sum_of_square  # FM interaction term
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)  # Optionally reduce to scalar
        return 0.5 * ix


class FMEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        """
        Parameters:
        - input_dim: Number of input features (e.g., columns in tabular data).
        - embed_dim: Dimension of embeddings for FM interactions.
        """
        super().__init__()
        self.indim = input_dim
        self.embed_dim = embed_dim
        self.outdim = 2 * embed_dim
        self.embedding = nn.Linear(self.indim, self.indim * self.embed_dim, bias=False)
        self.linear = FeaturesLinear(self.indim)
        self.fm = FactorizationMachine(reduce_sum=False)

    def forward(self, x, delta):
        x_embedding = self.embedding(x).reshape(-1, self.indim, self.embed_dim)  # Reshape to [B, d, k]
        delta_embedding = self.embedding(delta).reshape(-1, self.indim, self.embed_dim)  # Reshape to [B, d, k]
        embeddings = torch.cat((x_embedding, delta_embedding), dim=-1) # [B, d, 2k]

        # Linear component
        linear_part = self.linear(x)

        # FM interaction component
        fm_part = self.fm(embeddings)

        return torch.sigmoid(linear_part + fm_part).squeeze(1)