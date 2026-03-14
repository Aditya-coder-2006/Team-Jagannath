import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (SequenceLength, Batch, EmbeddingDim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TemporalTransformer(nn.Module):
    """
    Multi-head self attention over frame embeddings to provide temporal context.
    """
    def __init__(self, d_model: int = 1024, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: FloatTensor of shape (Batch, SequenceLength, EmbeddingDim)
        """
        # Convert to (Seq, Batch, Embed) for positional encoding
        src_seq_first = src.transpose(0, 1) * math.sqrt(self.d_model)
        src_seq_first = self.pos_encoder(src_seq_first)
        
        # Convert back to (Batch, Seq, Embed) for modern batch_first=True transformer
        src_batch_first = src_seq_first.transpose(0, 1)
        
        output = self.transformer_encoder(src_batch_first)
        return output
