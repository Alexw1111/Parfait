import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True,
            dim_feedforward=model_dim * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, SeqLen, InputDim]
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        # Permute for pooling: [Batch, ModelDim, SeqLen]
        x = x.permute(0, 2, 1)
        # Pool and squeeze: [Batch, ModelDim]
        x = self.output_pooling(x).squeeze(-1)
        return x