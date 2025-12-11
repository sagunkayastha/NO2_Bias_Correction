import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_prob=0.3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        # Use InstanceNorm instead of BatchNorm for batch-independent behavior
        self.bn1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.InstanceNorm1d(out_channels, affine=True)

        # Residual connection adjustment
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out


class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.fc2d = nn.Linear(hidden_dim, hidden_dim)
        self.fc3d = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x_2d, x_3d):
        """
        x_2d: (batch, hidden_dim)
        x_3d: (batch, hidden_dim)
        """
        attn_2d = self.attn(torch.tanh(self.fc2d(x_2d)))
        attn_3d = self.attn(torch.tanh(self.fc3d(x_3d)))
        attn_weights = torch.softmax(torch.cat([attn_2d, attn_3d], dim=1), dim=1)
        fused = attn_weights[:, 0:1] * x_2d + attn_weights[:, 1:2] * x_3d
        return fused


class AMFPredictor_res(nn.Module):
    def __init__(
        self,
        input_dim_2d,
        input_dim_3d,
        num_layers_3d=3,
        hidden_dim=64,
        dropout_prob=0.3,
    ):
        super(AMFPredictor_res, self).__init__()

        # 2D Feature Branch (using LayerNorm for batch-independent behavior)
        self.fc_2d = nn.Sequential(
            nn.Linear(input_dim_2d, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob),
        )

        # 3D Feature Branch with Residual Blocks
        layers = []
        in_channels = input_dim_3d
        for _ in range(num_layers_3d):
            layers.append(
                ResidualBlock1D(in_channels, hidden_dim, dropout_prob=dropout_prob)
            )
            in_channels = hidden_dim
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv3d = nn.Sequential(*layers)

        # Attention Fusion Layer
        self.attn_fusion = AttentionFusion(hidden_dim)

        # Fully Connected Final Prediction
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensures AMF is non-negative
        )

    def forward(self, x_2d, x_3d):
        """
        x_2d: Tensor of shape (batch, input_dim_2d)
        x_3d: Tensor of shape (batch, input_dim_3d, layers)
        """
        x_2d_out = self.fc_2d(x_2d)  # Process 2D features
        x_3d_out = self.conv3d(x_3d).squeeze(-1)  # Process 3D features

        # Attention-based fusion of 2D and 3D branches
        x = self.attn_fusion(x_2d_out, x_3d_out)

        return self.fc_final(x)
