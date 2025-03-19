import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLinearFeatureGatedFusion(nn.Module):
    def __init__(self, feature_dim):
        """
        Non-linear feature gated fusion module for RGB and NIR features.

        Args:
            feature_dim (int): Dimensionality of input features (RGB and NIR).
        """
        super(NonLinearFeatureGatedFusion, self).__init__()
        self.feature_dim = feature_dim

        # MLP layers for RGB and NIR features
        self.mlp_rgb = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        self.mlp_nir = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

        # 1D Average Pooling and Normalization
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.normalization = nn.BatchNorm1d(3 * feature_dim)

    def forward(self, t_r, t_n):
        """
        Forward pass for the non-linear feature gated fusion module.

        Args:
            t_r (torch.Tensor): RGB features of shape (batch_size, feature_dim).
            t_n (torch.Tensor): NIR features of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Fused features of shape (batch_size, 3 * feature_dim).
        """
        # Estimate feature vectors using MLP
        t_r_m = self.mlp_rgb(t_r) if t_r is not None else None
        t_n_m = self.mlp_nir(t_n) if t_n is not None else None

        # Decision gates for feature fusion
        if t_r is not None and t_n is not None:
            t_rn = torch.kron(t_r, t_n)  # Kronecker product
        elif t_r is None and t_n is not None:
            t_rn = torch.kron(t_r_m, t_n)
        elif t_r is not None and t_n is None:
            t_rn = torch.kron(t_r, t_n_m)
        else:
            raise ValueError("Both RGB and NIR features cannot be None.")

        # Flatten the Kronecker product output
        t_rn_flatten = t_rn.view(t_rn.size(0), -1)  # Flatten to 1D

        # Concatenate features
        if t_r is not None and t_n is not None:
            t_concat = torch.cat([t_r, t_rn_flatten, t_n], dim=1)
        elif t_r is None:
            t_concat = torch.cat([t_r_m, t_rn_flatten, t_n], dim=1)
        elif t_n is None:
            t_concat = torch.cat([t_r, t_rn_flatten, t_n_m], dim=1)

        # Apply 1D Average Pooling and Normalization
        t_concat = t_concat.unsqueeze(2)  # Add dimension for pooling
        t_concat = self.avg_pool(t_concat).squeeze(2)
        t_concat = self.normalization(t_concat)

        return t_concat


# Example usage
if __name__ == "__main__":
    feature_dim = 64  # Example feature dimension
    model = NonLinearFeatureGatedFusion(feature_dim)

    # Test case 1: Both RGB and NIR features are available
    t_r = torch.randn(10, feature_dim)  # RGB features
    t_n = torch.randn(10, feature_dim)  # NIR features
    output = model(t_r, t_n)
    print("Test case 1 output shape:", output.shape)  # Expected: (10, 192)

    # Test case 2: RGB features are missing
    t_r = None
    t_n = torch.randn(10, feature_dim)
    output = model(t_r, t_n)
    print("Test case 2 output shape:", output.shape)  # Expected: (10, 192)

    # Test case 3: NIR features are missing
    t_r = torch.randn(10, feature_dim)
    t_n = None
    output = model(t_r, t_n)
    print("Test case 3 output shape:", output.shape)  # Expected: (10, 192)