"""
MSOPT PyTorch Engine: 2D Scale-Time Spatial Conv-Transformer Backbone
====================================================================
PyTorch implementation of the MSOPT deep neural architecture:
1. 2D Scale-Time Spatial Grid Embedder (Token + Scale + Position Embeddings)
2. 2D Spatial Convolutional Feature Extractor (Inter-Scale Pattern Composition)
3. Multi-Head Self-Attention Transformer Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSOPT2DSpatialEmbedder(nn.Module):
    """
    Maps 2D Spatial Token Index Grid [Batch, N_Scales, Time_Steps] into 
    Dense 2D Spatial Embedding Tensor [Batch, Embed_Dim, N_Scales, Time_Steps].
    """
    def __init__(
        self,
        vocab_size: int,
        n_scales: int,
        max_time_steps: int = 1000,
        embed_dim: int = 64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_scales = n_scales
        self.embed_dim = embed_dim
        
        # 1. Codebook Token Embedding
        self.token_embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        # 2. Scale Positional Embedding (Y-axis)
        self.scale_embed = nn.Embedding(n_scales, embed_dim)
        
        # 3. Temporal Positional Embedding (X-axis)
        self.time_embed = nn.Embedding(max_time_steps, embed_dim)

    def forward(self, grid_indices: torch.Tensor) -> torch.Tensor:
        """
        Input: grid_indices [B, N_scales, T]
        Output: dense_embeddings [B, embed_dim, N_scales, T]
        """
        B, K, T = grid_indices.shape
        
        # Token embeddings: [B, K, T, D]
        tok_e = self.token_embed(grid_indices)
        
        # Scale embeddings: [K, D] -> [1, K, 1, D]
        scale_ids = torch.arange(K, device=grid_indices.device).unsqueeze(0).unsqueeze(2)
        scale_e = self.scale_embed(scale_ids)
        
        # Time embeddings: [T, D] -> [1, 1, T, D]
        time_ids = torch.arange(T, device=grid_indices.device).unsqueeze(0).unsqueeze(0)
        time_e = self.time_embed(time_ids)
        
        # Combined embedding: [B, K, T, D]
        embed = tok_e + scale_e + time_e
        
        # Permute to 2D CNN format: [B, D, K, T]
        return embed.permute(0, 3, 1, 2)


class MSOPTSpatialConvBlock(nn.Module):
    """
    2D Spatial Convolution Block capturing intra-scale time dynamics and inter-scale interactions.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        # Multi-scale 2D kernel convolutions
        self.conv_1x3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, 3), padding=(0, 1))
        self.conv_3x3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), padding=(1, 1))
        self.conv_3x1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 1), padding=(1, 0))
        
        self.norm = nn.BatchNorm2d(embed_dim * 3)
        self.project = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input/Output: [B, D, K, T]"""
        out1 = self.conv_1x3(x)
        out2 = self.conv_3x3(x)
        out3 = self.conv_3x1(x)
        
        concat = torch.cat([out1, out2, out3], dim=1)
        normed = self.norm(concat)
        proj = self.project(normed)
        return self.act(x + proj) # Residual connection


class MSOPTDeepNeuralClassifier(nn.Module):
    """
    Full Deep Neural Model: 2D Spatial Grid Embedder + 2D Conv Backbone + Transformer Encoder.
    """
    def __init__(
        self,
        vocab_size: int,
        n_scales: int,
        num_classes: int = 3,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2
    ):
        super().__init__()
        self.embedder = MSOPT2DSpatialEmbedder(vocab_size, n_scales, embed_dim=embed_dim)
        self.conv_block = MSOPTSpatialConvBlock(embed_dim=embed_dim)
        
        # Transformer Encoder over time steps
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * n_scales,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Head classifier
        self.head = nn.Sequential(
            nn.Linear(embed_dim * n_scales, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, grid_indices: torch.Tensor) -> torch.Tensor:
        """
        Input: [Batch, N_Scales, Time_Steps]
        Output: Class Logits [Batch, Num_Classes]
        """
        B, K, T = grid_indices.shape
        
        # 1. 2D Embeddings: [B, D, K, T]
        x_2d = self.embedder(grid_indices)
        
        # 2. 2D Spatial Convolution: [B, D, K, T]
        feat_2d = self.conv_block(x_2d)
        
        # 3. Reshape for Transformer: flatten scale and channel dims -> [B, T, D * K]
        feat_flat = feat_2d.permute(0, 3, 1, 2).reshape(B, T, -1)
        
        # 4. Transformer Attention across time
        trans_out = self.transformer(feat_flat)
        
        # 5. Pool last time step representation
        last_step = trans_out[:, -1, :]
        
        # 6. Classification Logits
        return self.head(last_step)


if __name__ == "__main__":
    B, K, T = 8, 12, 60
    vocab_size = 500
    
    dummy_grid = torch.randint(1, vocab_size, (B, K, T))
    model = MSOPTDeepNeuralClassifier(vocab_size=vocab_size, n_scales=K, num_classes=3)
    
    logits = model(dummy_grid)
    print(f"[MSOPT Neural Engine Check]")
    print(f"  Input 2D Grid Tensor Shape: {dummy_grid.shape}")
    print(f"  Output Logits Shape: {logits.shape} (Expected: [{B}, 3])")
