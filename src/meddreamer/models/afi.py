import torch
import torch.nn as nn


class PerFeatureLinearProjection(nn.Module):
    """
    Implements a per-feature linear projection from [B, D] to [B, D, k].

    For each feature d:
        e_d = x_d * W_d + b_d
    where W_d, b_d are learned separately for each feature.
    """
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)  # [D, k]
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))            # [D, k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]

        Returns:
            [B, D, k]
        """
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FactorizationMachine(nn.Module):
    """
    FM interaction term over the feature dimension.

    Input:
        x: [B, D, E]

    Output:
        [B, E]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_of_sum = torch.sum(x, dim=1) ** 2   # [B, E]
        sum_of_square = torch.sum(x ** 2, dim=1)   # [B, E]
        return 0.5 * (square_of_sum - sum_of_square)


class AFIEmbedding(nn.Module):
    """
    Adaptive Feature Integration module following the paper:

        E^(o)_t = W_o o_t       in R^{D x k}
        E^(Δ)_t = W_Δ Δ_t       in R^{D x k}

        E_t = [E^(o)_t | E^(Δ)_t]   in R^{D x 2k}

        FM(E_t) = 1/2 * [ (sum_j e_j)^2 - sum_j (e_j^2) ]   in R^{2k}

        \tilde{o}_t = sigmoid( Linear(o_t) + FM(E_t) )      in R^{2k}
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.outdim = 2 * embed_dim

        # Separate linear projections for observations and deltas
        self.obs_projection = PerFeatureLinearProjection(input_dim, embed_dim)
        self.delta_projection = PerFeatureLinearProjection(input_dim, embed_dim)

        # Final linear term uses only o_t, exactly as in the paper
        self.linear_obs = nn.Linear(input_dim, 2 * embed_dim)

        self.fm = FactorizationMachine()

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, D]  observed feature values at time t
            delta: [B, D]  per-feature time intervals at time t

        Returns:
            processed observation: [B, 2k]
        """
        # E^(o)_t in R^{B x D x k}
        obs_emb = self.obs_projection(x)

        # E^(Δ)_t in R^{B x D x k}
        delta_emb = self.delta_projection(delta)

        # E_t = [E^(o)_t | E^(Δ)_t] in R^{B x D x 2k}
        joint_emb = torch.cat([obs_emb, delta_emb], dim=-1)

        # FM(E_t) in R^{B x 2k}
        fm_term = self.fm(joint_emb)

        # Linear(o_t) in R^{B x 2k}
        linear_term = self.linear_obs(x)

        # \tilde{o}_t in R^{B x 2k}
        out = torch.sigmoid(linear_term + fm_term)
        return out