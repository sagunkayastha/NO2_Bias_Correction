import torch.nn as nn
from neuralop.models import FNO
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
        x = self.fno(x_3d)
        x = x.mean(dim=-1)
        return x


class MLP2DBranch(nn.Module):
    def __init__(self, input_dim_2d, hidden_dim, dropout_prob=0.3):
        super(MLP2DBranch, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim_2d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
        )
        self.mlp.apply(init_weights)

    def forward(self, x_2d):
        return self.mlp(x_2d)  # (batch, hidden_dim)


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=2):
        super(CrossAttentionFusion, self).__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_dim = hidden_dim

    def forward(self, x_2d, x_3d):
        """
        x_2d, x_3d: (batch, hidden_dim)
        """
        q = self.q_proj(x_2d).unsqueeze(1)  # (batch, 1, hidden_dim)
        k = self.k_proj(x_3d).unsqueeze(1)  # (batch, 1, hidden_dim)
        v = self.v_proj(x_3d).unsqueeze(1)  # (batch, 1, hidden_dim)

        scores = (q @ k.transpose(-2, -1)) / (self.hidden_dim**0.5)  # (batch, 1, 1)
        attn = self.softmax(scores)  # Still (batch, 1, 1)
        context = attn @ v  # (batch, 1, hidden_dim)
        context = context.squeeze(1)

        return self.out_proj(context + x_2d)  # Residual connection


class AMFPredictor_NoTransformer(nn.Module):
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
        super(AMFPredictor_NoTransformer, self).__init__()

        self.fc_2d = MLP2DBranch(input_dim_2d, hidden_dim, dropout_prob)

        self.fno3d = FNO(
            n_modes=(n_modes,),
            in_channels=input_dim_3d,
            out_channels=hidden_dim,
            hidden_channels=fno_hidden,
            projection_channel_ratio=2,
        )

        # self.attn_fusion = nn.Sequential(  # Replacing attention with simple fusion
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_prob),
        # )
        self.attn_fusion = MultiHeadAttentionFusion(hidden_dim, num_heads=n_heads)
        # self.attn_fusion = CrossAttentionFusion(hidden_dim)

        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(beta=2),
        )

    def forward(self, x_2d, x_3d):
        x_2d_out = self.fc_2d(x_2d)
        x_3d_out = self.fno3d(x_3d).mean(dim=-1)

        x = self.attn_fusion(x_2d_out, x_3d_out)
        # x = torch.cat([x_2d_out, x_3d_out], dim=1)
        # x = self.attn_fusion(x)
        x = self.fc_final(x)
        return x


class MLP3DBranch(nn.Module):
    def __init__(self, input_dim_3d, hidden_dim, dropout_prob=0.3):
        super(MLP3DBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim_3d, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x_3d):
        # x_3d: (batch, channels, layers)
        x = x_3d.mean(dim=-1)  # Global average over layers
        return self.mlp(x)  # (batch, hidden_dim)


class AMFPredictor_NoFNO(nn.Module):
    def __init__(self, input_dim_2d, input_dim_3d, hidden_dim=64, dropout_prob=0.3):
        super(AMFPredictor_NoFNO, self).__init__()

        self.fc_2d = Transformer2DBranch(
            input_dim_2d, hidden_dim, num_layers=2, dropout_prob=dropout_prob
        )
        self.fc_3d = MLP3DBranch(input_dim_3d, hidden_dim, dropout_prob)

        self.attn_fusion = MultiHeadAttentionFusion(hidden_dim, num_heads=2)

        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(beta=2),
        )

    def forward(self, x_2d, x_3d):
        x_2d_out = self.fc_2d(x_2d)
        x_3d_out = self.fc_3d(x_3d)
        x = self.attn_fusion(x_2d_out, x_3d_out)
        x = self.fc_final(x)
        return x
