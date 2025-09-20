# models/transformer_model_v2.py
import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, d_model=128, n_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, n_features)
        
    def forward(self, x):
        # x: [batch_size, seq_len, n_features]
        z = self.embedding(x) + self.positional_encoding
        memory = self.encoder(z)
        decoded = self.decoder(z, memory)
        out = self.output_layer(decoded)
        return out
