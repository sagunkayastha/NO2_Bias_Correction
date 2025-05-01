import torch
import torch.nn as nn
from neuralop.models import FNO


### 🔹 Transformer-Enhanced 2D Feature Branch
class Transformer2DBranch(nn.Module):
    def __init__(self, input_dim_2d, hidden_dim, num_layers=2, dropout_prob=0.3):
        super(Transformer2DBranch, self).__init__()
        self.embedding = nn.Linear(input_dim_2d, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_prob,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x_2d):
        """
        x_2d: (batch, input_dim_2d)
        """
        x = self.embedding(x_2d).unsqueeze(1)  # (batch, 1, hidden_dim)
        x = self.transformer_encoder(x).squeeze(1)  # (batch, hidden_dim)
        return x


### 🔹 Multi-Head Attention Fusion
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=2):
        super(MultiHeadAttentionFusion, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_2d, x_3d):
        """
        x_2d: (batch, hidden_dim)
        x_3d: (batch, hidden_dim)
        """
        q = self.query(x_2d)
        k = self.key(x_3d)
        v = self.value(x_3d)

        attention_scores = self.softmax(
            q @ k.transpose(-2, -1) / (x_3d.size(-1) ** 0.5)
        )
        fused = attention_scores @ v + x_2d  # Residual connection
        return fused


### 🔹 FNO-based 3D Feature Branch
class FNO3DBranch(nn.Module):
    def __init__(self, input_dim_3d, hidden_dim, n_modes=16, fno_hidden=32):
        super(FNO3DBranch, self).__init__()
        self.fno = FNO(
            n_modes=(n_modes,),
            in_channels=input_dim_3d,
            out_channels=hidden_dim,
            hidden_channels=fno_hidden,
            projection_channel_ratio=2,
        )

    def forward(self, x_3d):
        """
        x_3d: (batch, input_dim_3d, height, width)
        """
        x = self.fno(x_3d)  # Output shape: (batch, hidden_dim, layers)
        x = x.mean(dim=-1)  # Average over the 'layers' dimension
        return x


# 🔹 Full AMF Predictor Model
class AMFPredictor_FNO(nn.Module):
    def __init__(
        self,
        input_dim_2d,
        input_dim_3d,
        hidden_dim=64,
        n_modes=16,
        fno_hidden=32,
        n_heads=2,
        dropout_prob=0.3,
    ):
        super(AMFPredictor_FNO, self).__init__()

        # 🔹 Transformer-based 2D Feature Extractor
        self.fc_2d = Transformer2DBranch(
            input_dim_2d, hidden_dim, num_layers=2, dropout_prob=dropout_prob
        )

        # 🔹 FNO-based 3D Feature Extractor
        self.fno3d = FNO3DBranch(input_dim_3d, hidden_dim, n_modes, fno_hidden)

        # 🔹 Multi-Head Attention Fusion Layer
        self.attn_fusion = MultiHeadAttentionFusion(hidden_dim, num_heads=n_heads)

        # 🔹 Fully Connected Output (Ensuring Non-Negative AMF)
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(beta=2),  # AMF must be non-negative
        )
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_2d, x_3d):
        x_2d_out = self.fc_2d(x_2d)  # Transformer for 2D inputs
        x_3d_out = self.fno3d(x_3d)  # FNO processes 3D inputs

        # 🔹 Multi-Head Attention Fusion of 2D & 3D Outputs
        x = self.attn_fusion(x_2d_out, x_3d_out)
        x = self.fc_final(x)  # removed sequential.
        # x = x + self.bias #added bias.
        return x
